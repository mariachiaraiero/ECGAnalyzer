import streamlit as st
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import os
import sys
import tempfile
import wfdb
from scipy.interpolate import CubicSpline

# Aggiungo i percorsi per importare i modelli e il post-processing
import test_ensemble_fasi_post
from test_ensemble_fasi_post import EnhancedUNetV3, ECGUNetFase6, postprocess_mask, apply_ecg_filters

import eval_ensemble_ludb
from eval_ensemble_ludb import postprocess_mask_multibeat

import picchi_final
from picchi_final import find_all_peaks, get_matched_classes, _mark_peak

import ludb_parser

# --- Costanti ---
TARGET_FS = 500
MEDIAN_LEN = 600
CONT_LEN = 5000
LEADS_ORDER = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

C_P   = '#2196F3'
C_QRS = '#E53935'
C_T   = '#43A047'
C_ST  = '#FDD835'

st.set_page_config(layout="wide", page_title="ECG Ensemble Viewer")

# --- Logica di Parsing ---
def resample_signal(signal, original_fs, target_fs=500):
    if original_fs == target_fs:
        return signal
    n = len(signal)
    T = n / original_fs
    t = np.array([(2*i-1)*T/(2*n) for i in range(1,n+1)])
    cs = CubicSpline(t, signal)
    m = int(target_fs * T)
    t_new = np.array([(2*i-1)*T/(2*m) for i in range(1,m+1)])
    return cs(t_new).astype(np.float32)
def parse_ge_xml(xml_content):
    tree = ET.parse(io.BytesIO(xml_content))
    root = tree.getroot()
    
    ns = {"ns": "urn:ge:sapphire:dcar_1"}
    median_wfs = root.findall(".//ns:medianTemplate//ns:ecgWaveform", namespaces=ns)
    all_wfs = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)
    if not median_wfs and not all_wfs:
        ns = {"ns": "urn:ge:sapphire:sapphire_3"}
        median_wfs = root.findall(".//ns:medianTemplate//ns:ecgWaveform", namespaces=ns)
        all_wfs = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)

    # Estrai Median Signals
    median_signals = {}
    for wf in median_wfs:
        lead = wf.get('lead')
        if not lead: continue
        vals = [int(v) for v in wf.get('V', '').split()]
        if not vals: continue
        
        # Gestione del bug hardware GE sui median beats (-32768)
        clean_vals = []
        last_valid = 0
        for v in vals:
            if v != -32768:
                clean_vals.append(v)
                last_valid = v
            else:
                clean_vals.append(last_valid)
                
        if len(clean_vals) > MEDIAN_LEN: clean_vals = clean_vals[:MEDIAN_LEN]
        elif len(clean_vals) < MEDIAN_LEN: clean_vals += [last_valid] * (MEDIAN_LEN - len(clean_vals))
        
        median_signals[lead] = np.array([4.88 * v / 1000.0 for v in clean_vals], dtype=np.float32)

    # Estrai Continuous Signals
    cont_signals = {}
    strip_wfs = [wf for wf in all_wfs if wf.get('asizeVT','0') != '600']
    for wf in strip_wfs:
        lead = wf.get('lead', '').upper()
        vt_data = wf.get('V', '')
        if not lead or not vt_data: continue
        try:
            values = [(4.88 * int(val) / 1000) for val in vt_data.split()]
            cont_signals[lead] = np.array(values, dtype=np.float32)
        except ValueError:
            pass

    # Annotazioni Globali
    global_ann = {}
    g_node = root.find(".//ns:medianTemplate/ns:measurements/ns:global", namespaces=ns)
    if g_node is not None:
        for child in g_node:
            v = child.get('V')
            if v and v != '-32768':
                tag = child.tag.split('}')[-1]
                try:
                    global_ann[tag] = int(v)
                except ValueError: pass

    # Posizioni battiti (Timbro)
    beat_positions = []
    for beat in root.findall(".//ns:evt/ns:beat/ns:any", namespaces=ns):
        toc_node = beat.find("ns:tpoint/ns:toc", namespaces=ns)
        if toc_node is not None:
            toc_v = toc_node.get('V')
            if toc_v:
                try: beat_positions.append(int(toc_v))
                except ValueError: pass
    beat_positions.sort()

    # Sample rate continuo
    sample_rate = TARGET_FS
    for sr_node in root.findall(".//ns:ecgWaveformMXG/ns:sampleRate", namespaces=ns):
        sr_val = sr_node.get('V')
        if sr_val: sample_rate = int(sr_val)

    # Diciture diagnostiche
    statements = []
    for st_node in root.findall(".//ns:interpretation//ns:statement", namespaces=ns):
        txt = st_node.get('V')
        if txt:
            statements.append(txt)

    return median_signals, cont_signals, global_ann, beat_positions, sample_rate, statements

def parse_ludb_wfdb(tmpdir, record_name):
    record_path = os.path.join(tmpdir, record_name)
    record = wfdb.rdrecord(record_path)
    
    fs = record.fs
    sig_len = record.p_signal.shape[0]
    
    cont_sigs = {}
    gt_masks = {}
    
    for ci, sn in enumerate(record.sig_name):
        lead_name = sn.upper()
        sig = record.p_signal[:, ci].astype(np.float32)
        
        mask = np.zeros(sig_len, dtype=np.int64)
        try:
            ann = wfdb.rdann(record_path, sn.lower())
            symbols = ann.symbol
            samples = ann.sample
            n = len(symbols)
            PEAK_TO_CLASS = {'p': 1, 'N': 2, 't': 3}
            i = 0
            while i < n:
                sym = str(symbols[i]).strip()
                if sym == '(':
                    onset = int(samples[i])
                    w_class = None
                    offset = None
                    j = i + 1
                    while j < n:
                        sj = str(symbols[j]).strip()
                        if sj in PEAK_TO_CLASS: w_class = PEAK_TO_CLASS[sj]
                        elif sj == ')':
                            offset = int(samples[j])
                            break
                        j += 1
                    if w_class is not None and offset is not None:
                        mask[max(0, onset):min(sig_len-1, offset)+1] = w_class
                    if offset is not None: i = j + 1
                    else: i += 1
                else: i += 1
        except Exception:
            pass
            
        if fs != TARGET_FS:
            T = sig_len / fs
            t = np.array([(2 * i - 1) * T / (2 * sig_len) for i in range(1, sig_len + 1)])
            cs = CubicSpline(t, sig)
            m = int(TARGET_FS * T)
            t_new = np.array([(2 * i - 1) * T / (2 * m) for i in range(1, m + 1)])
            sig = cs(t_new)
            indices = np.round(np.linspace(0, sig_len - 1, m)).astype(int)
            mask = mask[indices]
            
        cont_sigs[lead_name] = sig
        gt_masks[lead_name] = mask
        
    return cont_sigs, gt_masks

def build_continuous_mask_from_ann(beat_positions, sample_rate, ann, target_len, matched=set()):
    mask = np.zeros(target_len, dtype=np.int64)
    
    if 'PACING' in matched:
        # Per PACING: le annotazioni GE rappresentano solo il tick di stimolazione elettrica.
        # Usiamo offset fisiologici fissi dal toc (spike) per ricostruire boundaries reali.
        # Valori tipici per un battito paceato ventricoale a freq normale:
        #   P (atrial activity/sensing): toc - 250ms -> toc - 80ms
        #   QRS (spike + complesso paceato): toc - 20ms -> toc + 160ms
        #   T (onda T paceata): toc + 160ms -> toc + 400ms
        for toc_native in sorted(beat_positions):
            toc_t = int(toc_native * TARGET_FS / sample_rate)
            ms = TARGET_FS / 1000.0  # campioni per ms
            p_s  = max(0, toc_t - int(250*ms));  p_e  = max(0, toc_t - int(80*ms))
            q_s  = max(0, toc_t - int(20*ms));   q_e  = min(target_len-1, toc_t + int(160*ms))
            t_s  = min(target_len-1, q_e);        t_e  = min(target_len-1, toc_t + int(400*ms))
            if p_s < p_e: mask[p_s:p_e+1] = 1
            if q_s < q_e: mask[q_s:q_e+1] = 2
            if t_s < t_e: mask[t_s:t_e+1] = 3
        return mask

    p_onset = ann.get('P_Onset'); p_offset = ann.get('P_Offset')
    q_onset = ann.get('Q_Onset'); q_offset = ann.get('Q_Offset')
    t_onset = ann.get('T_Onset'); t_offset = ann.get('T_Offset')

    if q_onset is None or not beat_positions:
        return mask

    anchor_ms = 500
    for toc_native in sorted(beat_positions):
        toc_t = int(toc_native * TARGET_FS / sample_rate)
        def proj(ann_ms): return toc_t + int((ann_ms - anchor_ms) * TARGET_FS / 1000)

        if p_onset is not None and p_offset is not None:
            ps = max(0, proj(p_onset)); pe = min(target_len - 1, proj(p_offset))
            if ps < pe: mask[ps:pe + 1] = 1

        if q_onset is not None and q_offset is not None:
            qs = max(0, proj(q_onset)); qe = min(target_len - 1, proj(q_offset))
            if qs < qe: mask[qs:qe + 1] = 2

        if q_offset is not None and t_onset is not None:
            ss = max(0, proj(q_offset)); se = min(target_len - 1, proj(t_onset))
            if ss < se: mask[ss:se + 1] = 4

        t_start = t_onset if t_onset is not None else q_offset
        if t_start is not None and t_offset is not None:
            ts = max(0, proj(t_start)); te = min(target_len - 1, proj(t_offset))
            if ts < te: mask[ts:te + 1] = 3

    return mask


def outward_search_boundaries(sig_med, anchor_s=250, fs=TARGET_FS):
    n = len(sig_med)
    edge_w = int(0.10 * fs)
    edges = np.concatenate([sig_med[:edge_w], sig_med[-edge_w:]])
    baseline = np.median(edges)
    noise_std = max(np.std(edges), 1e-6)
    thr = 2.5 * noise_std 
    ws = max(0, anchor_s - int(0.08*fs))
    we = min(n, anchor_s + int(0.08*fs))
    spike_idx = ws + int(np.argmax(np.abs(sig_med[ws:we])))
    start_idx = spike_idx
    w_back = int(0.02 * fs)
    for i in range(spike_idx, w_back, -1):
        if np.all(np.abs(sig_med[i-w_back:i] - baseline) < thr):
            start_idx = i; break
    end_idx = spike_idx
    w_fwd = int(0.05 * fs)
    for i in range(spike_idx, n - w_fwd):
        if np.all(np.abs(sig_med[i:i+w_fwd] - baseline) < thr):
            end_idx = i; break
    q_onset = max(start_idx, spike_idx - int(0.03*fs))
    j_point = min(end_idx, spike_idx + int(0.30*fs))
    srch_s = spike_idx + int(0.02*fs)
    if end_idx - srch_s > 10:
        d = np.abs(np.diff(sig_med[srch_s:end_idx]))
        sm = np.convolve(d, np.ones(9)/9, mode='same')
        j_thr = 0.15 * np.max(sm) if np.max(sm) > 0 else 1.0
        for i in range(len(sm)):
            if sm[i] < j_thr:
                j_point = srch_s + i; break
    ms = lambda s: round(s * 1000 / fs)
    return {'P_Onset': ms(start_idx), 'P_Offset': ms(q_onset),
            'Q_Onset': ms(q_onset), 'Q_Offset': ms(j_point),
            'T_Onset': ms(j_point), 'T_Offset': ms(end_idx)}

def get_corrected_pacing_ann(global_ann, median_sigs, matched):
    if 'PACING' in matched and median_sigs:
        ref_lead = 'II' if 'II' in median_sigs else list(median_sigs.keys())[0]
        sig_med = median_sigs[ref_lead]
        # Normalize as in inference
        sig_med = (sig_med - np.mean(sig_med)) / (np.std(sig_med) + 1e-8)
        return outward_search_boundaries(sig_med)
    return global_ann


def build_median_gt_mask(global_ann, fs=TARGET_FS):
    mask = np.zeros(MEDIAN_LEN, dtype=np.int64)
    def to_idx(ms): return int(ms * fs / 1000.0)
    
    p_onset = global_ann.get('P_Onset') or global_ann.get('POnset')
    p_offset = global_ann.get('P_Offset') or global_ann.get('POffset')
    q_onset = global_ann.get('Q_Onset') or global_ann.get('QOnset')
    q_offset = global_ann.get('Q_Offset') or global_ann.get('QOffset')
    t_offset = global_ann.get('T_Offset') or global_ann.get('TOffset')
    t_onset = global_ann.get('T_Onset') or q_offset

    if p_onset is not None and p_offset is not None:
        s, e = to_idx(p_onset), to_idx(p_offset)
        if s < e: mask[s:e+1] = 1
    if q_onset is not None and q_offset is not None:
        s, e = to_idx(q_onset), to_idx(q_offset)
        if s < e: mask[s:e+1] = 2
    if q_offset is not None and t_offset is not None:
        # GE convention: T starts at Q offset
        s, e = to_idx(q_offset), to_idx(t_offset)
        if s < e: mask[s:e+1] = 3
    return mask

def build_continuous_mask_from_ann(beat_positions, sample_rate, global_ann, target_len):
    mask = np.zeros(target_len, dtype=np.int64)
    fs = TARGET_FS
    
    p_onset = global_ann.get('P_Onset') or global_ann.get('POnset')
    p_offset = global_ann.get('P_Offset') or global_ann.get('POffset')
    q_onset = global_ann.get('Q_Onset') or global_ann.get('QOnset')
    q_offset = global_ann.get('Q_Offset') or global_ann.get('QOffset')
    t_offset = global_ann.get('T_Offset') or global_ann.get('TOffset')
    t_onset = global_ann.get('T_Onset') or q_offset

    for toc_native in sorted(beat_positions):
        toc_t = int(toc_native * fs / sample_rate)
        def proj(ms_val): return toc_t + int((ms_val - 500) * fs / 1000.0)
        
        if p_onset is not None and p_offset is not None:
            s, e = max(0, proj(p_onset)), min(target_len-1, proj(p_offset))
            if s < e: mask[s:e+1] = 1
        if q_onset is not None and q_offset is not None:
            s, e = max(0, proj(q_onset)), min(target_len-1, proj(q_offset))
            if s < e: mask[s:e+1] = 2
        if t_onset is not None and t_offset is not None:
            s, e = max(0, proj(t_onset)), min(target_len-1, proj(t_offset))
            if s < e: mask[s:e+1] = 3
    return mask

def stamp_median_mask(beat_positions, sample_rate, median_mask, target_len, anchor_ms=500):
    mask = np.zeros(target_len, dtype=np.int64)
    segs = extract_intervals_multibeat(median_mask)
    
    # Calcola anchor dal vero centro QRS nella maschera mediana predetta,
    # invece di usare un valore fisso 500ms che puo' essere sfasato.
    qrs_segs = segs[2]  # class 2 = QRS
    if qrs_segs:
        # Usa il centro del primo (unico) QRS nel battito medio
        q_on, q_off = qrs_segs[0]
        anchor_idx = (q_on + q_off) // 2
    else:
        # Fallback al valore fisso
        anchor_idx = int(anchor_ms * TARGET_FS / 1000.0)

    for toc_native in sorted(beat_positions):
        toc_t = int(toc_native * TARGET_FS / sample_rate)
        for c in [1, 2, 3, 4]:
            for (on_idx, off_idx) in segs[c]:
                rel_on = on_idx - anchor_idx
                rel_off = off_idx - anchor_idx
                s = max(0, toc_t + rel_on)
                e = min(target_len - 1, toc_t + rel_off)
                if s <= e:
                    mask[s:e+1] = c
    return mask

@st.cache_resource
def load_models():
    import glob
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict = {}
    
    pt_files = glob.glob("best_fase*.pt")
    
    for pt in pt_files:
        name = pt.split('.')[0].replace('best_', '').capitalize()
        if name == "Fase2_v3": name = "Fase2"
            
        model = EnhancedUNetV3(in_channels=1, num_classes=4, f=12).to(device)
        ckpt = torch.load(pt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        ckpt = {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k: v for k, v in ckpt.items()}
        
        try:
            model.load_state_dict(ckpt, strict=True)
            model.eval()
            models_dict[name] = model
        except Exception as e:
            print(f"Errore caricamento {pt}: {e}")
            
    return models_dict, device

# --- Funzioni di Plotting ---
def compute_median_beat_from_continuous(sig, mask, gt_mask=None, fs=500, median_len=600):
    """
    Estrae i battiti dal tracciato continuo usando la maschera predetta per allinearli sui QRS,
    e ne calcola la mediana. Se gt_mask è passata, calcola anche la maschera GT mediana (la moda).
    """
    # Evitiamo import circolari, la funzione apply_ecg_filters è già importata globalmente
    sig_filt = apply_ecg_filters(sig, fs=fs)
    qrs_list = extract_intervals_multibeat(mask).get(2, [])
    
    if not qrs_list:
        if gt_mask is not None: return None, None
        return None
        
    beats = []
    gt_beats = []
    margin_r = int(0.15 * fs)
    pre_samples = int(0.4 * fs)  # 400 ms prima
    post_samples = median_len - pre_samples # 800 ms dopo (se median_len=600)
    
    for q_on, q_off in qrs_list:
        center = (q_on + q_off) // 2
        ws = max(0, center - margin_r)
        we = min(len(sig_filt), center + margin_r + 1)
        if we <= ws: continue
        r_anchor = ws + int(np.argmax(np.abs(sig_filt[ws:we])))
        
        start_idx = r_anchor - pre_samples
        end_idx = r_anchor + post_samples
        
        valid_start = max(0, start_idx)
        valid_end = min(len(sig), end_idx)
        
        beat_start = max(0, -start_idx)
        beat_end = beat_start + (valid_end - valid_start)
        
        if valid_end > valid_start:
            beat = np.zeros(median_len, dtype=np.float32)
            beat[beat_start:beat_end] = sig[valid_start:valid_end]
            beats.append(beat)
            
            if gt_mask is not None:
                beat_gt = np.zeros(median_len, dtype=np.int64)
                beat_gt[beat_start:beat_end] = gt_mask[valid_start:valid_end]
                gt_beats.append(beat_gt)
            
    if not beats:
        if gt_mask is not None: return None, None
        return None
        
    med_sig = np.median(beats, axis=0)
    
    if gt_mask is not None:
        med_gt = np.apply_along_axis(lambda x: np.bincount(x, minlength=5).argmax(), axis=0, arr=np.array(gt_beats, dtype=np.int64))
        return med_sig, med_gt
        
    return med_sig


def extract_intervals_multibeat(mask):
    segs = {1: [], 2: [], 3: [], 4: []}
    for c in [1, 2, 3, 4]:
        is_c = (mask == c).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        for o, f in zip(on, off): segs[c].append((o, f))
        
    if not segs[4]:
        for q_on, q_off in segs[2]:
            t_candidates = [t_on for t_on, t_off in segs[3] if t_on > q_off]
            if t_candidates:
                t_on = min(t_candidates)
                if t_on - q_off < int(400 * TARGET_FS / 1000): 
                    segs[4].append((q_off, t_on))
    return segs

def compute_metrics_for_app(pred_mask, gt_mask, fs=TARGET_FS, tol_ms=150):
    tol = int(tol_ms / 1000.0 * fs)
    t_segs_dict = extract_intervals_multibeat(gt_mask)
    p_segs_dict = extract_intervals_multibeat(pred_mask)
    
    res = {c: {'onset': {'TP':0, 'FP':0, 'FN':0}, 'offset': {'TP':0, 'FP':0, 'FN':0}} for c in [1, 2, 3]}
    
    for c in [1, 2, 3]:
        t_segs = t_segs_dict[c]
        p_segs = p_segs_dict[c]
        for is_off, idx in [(False, 0), (True, 1)]:
            ptype = 'offset' if is_off else 'onset'
            t_pts = [s[idx] for s in t_segs]
            p_pts = [s[idx] for s in p_segs]
            matched = set()
            for tp in t_pts:
                valid = [pp for pp in p_pts if abs(pp - tp) <= tol and pp not in matched]
                if valid:
                    matched.add(min(valid, key=lambda pp: abs(pp - tp)))
                    res[c][ptype]['TP'] += 1
                else: 
                    res[c][ptype]['FN'] += 1
            res[c][ptype]['FP'] += len(p_pts) - len(matched)
            
    m, f1_list = {}, []
    for c in [1, 2, 3]:
        for pt in ['onset', 'offset']:
            d = res[c][pt]
            se = d['TP'] / (d['TP'] + d['FN'] + 1e-8)
            ppv = d['TP'] / (d['TP'] + d['FP'] + 1e-8)
            f1 = 2 * se * ppv / (se + ppv + 1e-8)
            m[f'{c}_{pt}_Se'] = se * 100
            m[f'{c}_{pt}_PPV'] = ppv * 100
            m[f'{c}_{pt}_F1'] = f1 * 100
            f1_list.append(f1)
    
    m['f1_p'] = (m['1_onset_F1'] + m['1_offset_F1']) / 2
    m['se_p'] = (m['1_onset_Se'] + m['1_offset_Se']) / 2
    m['ppv_p'] = (m['1_onset_PPV'] + m['1_offset_PPV']) / 2

    m['f1_q'] = (m['2_onset_F1'] + m['2_offset_F1']) / 2
    m['se_q'] = (m['2_onset_Se'] + m['2_offset_Se']) / 2
    m['ppv_q'] = (m['2_onset_PPV'] + m['2_offset_PPV']) / 2

    m['f1_t'] = (m['3_onset_F1'] + m['3_offset_F1']) / 2
    m['se_t'] = (m['3_onset_Se'] + m['3_offset_Se']) / 2
    m['ppv_t'] = (m['3_onset_PPV'] + m['3_offset_PPV']) / 2

    m['f1_macro'] = np.mean(f1_list) * 100
    
    return m

def shade_segs(ax, segs, show_st=True):
    classes = [(1, C_P, 'P-wave'), (2, C_QRS, 'QRS'), (3, C_T, 'T-wave')]
    if show_st:
        classes.insert(2, (4, C_ST, 'ST'))
        
    for c, color, lbl in classes:
        added = False
        for (on, off) in segs[c]:
            s_ms = on * 1000 / TARGET_FS
            e_ms = off * 1000 / TARGET_FS
            ax.axvspan(s_ms, e_ms, alpha=0.35, color=color, label=lbl if not added else "")
            added = True


def plot_single_median_beat(lead, sig, gt_mask_base, models, matched, fs=TARGET_FS, show_peaks=False, model_choice="Ensemble", show_gt_st=False):
    models_dict, device = models
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.5), sharex=True, gridspec_kw={'wspace': 0.1})
    ax_gt, ax_pred = axes[0], axes[1]
    
    ax_gt.set_title(f"GT Ground Truth - {lead}", fontsize=10)
    ax_pred.set_title(f"Predizione {model_choice} + Picchi - {lead}", fontsize=10)
    ax_gt.set_ylabel(lead, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
    
    t_ms = np.arange(MEDIAN_LEN) * 1000 / fs
    
    # 1. Inferenza
    sig_norm = ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float32)
    sig_pad = np.pad(sig_norm, (4, 4), mode='constant')
    x = torch.tensor(sig_pad).unsqueeze(0).unsqueeze(0).to(device)
    
    pred_mask = np.zeros(MEDIAN_LEN, dtype=np.int64)
    with torch.no_grad():
        if model_choice.startswith("Ensemble 2+5+6"):
            model_p = models_dict.get("Fase2")
            model_qrs = models_dict.get("Fase5")
            model_t = models_dict.get("Fase6")
            p_p = torch.nn.functional.softmax(model_p(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_p else np.zeros(MEDIAN_LEN)
            p_qrs = torch.nn.functional.softmax(model_qrs(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_qrs else np.zeros(MEDIAN_LEN)
            p_t = torch.nn.functional.softmax(model_t(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_t else np.zeros(MEDIAN_LEN)
            pred_mask[p_t == 3] = 3
            pred_mask[p_p == 1] = 1
            pred_mask[p_qrs == 2] = 2
        elif model_choice.startswith("Ensemble 2+5+2"):
            model_p = models_dict.get("Fase2")
            model_qrs = models_dict.get("Fase5")
            model_t = models_dict.get("Fase2")
            p_p = torch.nn.functional.softmax(model_p(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_p else np.zeros(MEDIAN_LEN)
            p_qrs = torch.nn.functional.softmax(model_qrs(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_qrs else np.zeros(MEDIAN_LEN)
            p_t = torch.nn.functional.softmax(model_t(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model_t else np.zeros(MEDIAN_LEN)
            pred_mask[p_t == 3] = 3
            pred_mask[p_p == 1] = 1
            pred_mask[p_qrs == 2] = 2
            
            # Remove ST gap by extending QRS wave to touch T onset
            last_2 = -1
            for i in range(len(pred_mask)):
                if pred_mask[i] == 2: last_2 = i
                elif pred_mask[i] == 3:
                    if last_2 != -1: pred_mask[last_2+1:i] = 2
                    last_2 = -1
                elif pred_mask[i] == 1: last_2 = -1
        else:
            target_name = model_choice.replace("Solo ", "")
            model = models_dict.get(target_name)
            pm = torch.nn.functional.softmax(model(x), dim=1).squeeze(0).cpu()[:, 4:604].argmax(0).numpy() if model else np.zeros(MEDIAN_LEN)
            pred_mask[pm == 2] = 2
            pred_mask[pm == 1] = 1
            pred_mask[pm == 3] = 3

    pred_mask = postprocess_mask(pred_mask, fs=fs)
    
    # 2. Trova picchi
    peaks_dict = {}
    if show_peaks:
        from picchi_final import extract_intervals
        sig_for_peaks = apply_ecg_filters(sig, fs=fs)
        boundaries = extract_intervals(pred_mask, fs=fs)
        peaks_dict = find_all_peaks(sig_for_peaks, boundaries, matched, lead_name=lead, fs=fs)
    
    # Plot GT
    ax_gt.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
    shade_segs(ax_gt, extract_intervals_multibeat(gt_mask_base), show_st=show_gt_st)
    
    # Plot Pred
    ax_pred.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
    shade_segs(ax_pred, extract_intervals_multibeat(pred_mask), show_st=True)
    
    # Mark peaks
    if show_peaks:
        for p_name in ['Q', 'R', 'S', 'T', 'J']:
            if p_name == 'J' and not (lead.upper() == 'AVR' or lead.upper().startswith('V')):
                continue
            pos_x = peaks_dict.get(p_name)
            if pos_x is not None:
                _mark_peak(ax_pred, sig, t_ms, pos_x, p_name, fs=fs, show_text=True)
                
        if peaks_dict.get("is_p_bifasica"):
            _mark_peak(ax_pred, sig, t_ms, peaks_dict.get("P_prime"), "P", fs=fs, show_text=True)
        else:
            _mark_peak(ax_pred, sig, t_ms, peaks_dict.get("P"), "P", fs=fs, show_text=True)
            
        if peaks_dict.get("T_bifasica_secondaria") is not None:
            idx = peaks_dict["T_bifasica_secondaria"]
            ax_pred.scatter([t_ms[idx]], [sig[idx]], marker="^", color="#81C784",
                            s=30, zorder=8, edgecolors="white", linewidths=0.4)
            
    # Formattazione
    for ax in [ax_gt, ax_pred]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    
    return fig


def generate_plot(sig, gt_mask, pred_pre, pred_post, title_mode, lead_name, show_peaks=False, matched=set(), show_text=True, model_choice="Ensemble", show_gt_st=False):
    t_ms = np.arange(len(sig)) * 1000 / TARGET_FS

    gt_segs = extract_intervals_multibeat(gt_mask)
    pre_segs = extract_intervals_multibeat(pred_pre)
    post_segs = extract_intervals_multibeat(pred_post)

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={'hspace': 0.4})
    
    # 1. Ground Truth
    axes[0].plot(t_ms, sig, color='k', lw=1)
    shade_segs(axes[0], gt_segs, show_st=show_gt_st)
    axes[0].set_title(f"{title_mode} - Ground Truth", loc='left', color='#555', fontsize=10)
    axes[0].set_ylabel(lead_name, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
    axes[0].legend(loc='upper right', framealpha=1.0)
    axes[0].grid(alpha=0.3)

    # 2. Pre-Processing
    axes[1].plot(t_ms, sig, color='k', lw=1)
    shade_segs(axes[1], pre_segs, show_st=True)
    axes[1].set_title(f"2. {model_choice} Diretto (Senza Post-Processing)", loc='left', color='#555', fontsize=10)
    axes[1].legend(loc='upper right', framealpha=1.0)
    axes[1].grid(alpha=0.3)

    # 3. Post-Processing + Picchi
    axes[2].plot(t_ms, sig, color='k', lw=1)
    shade_segs(axes[2], post_segs, show_st=True)
    
    if show_peaks:
        p_list = post_segs.get(1, [])
        qrs_list = post_segs.get(2, [])
        t_list = post_segs.get(3, [])
        sig_filt = apply_ecg_filters(sig, fs=TARGET_FS)
        margin_r = int(0.15 * TARGET_FS)   # ±150ms per cercare il vero R
        
        for q_on, q_off in qrs_list:
            # 1. Centro del rettangolo QRS stampato
            center = (q_on + q_off) // 2
            
            # 2. Vero picco R: massima escursione assoluta dal baseline in ±150ms
            ws = max(0, center - margin_r)
            we = min(len(sig_filt), center + margin_r + 1)
            r_anchor = ws + int(np.argmax(np.abs(sig_filt[ws:we])))
            
            # 3. Finestra QRS reale: simmetrica attorno al vero R, larga quanto la stampata
            half_qrs = max((q_off - q_on) // 2, int(0.05 * TARGET_FS)) + 5
            real_q_on  = max(0, r_anchor - half_qrs)
            real_q_off = min(len(sig) - 1, r_anchor + half_qrs)
            
            # 4. Associa P precedente e T successiva al vero QRS
            p_on, p_off = None, None
            p_cands = [(po, pf) for po, pf in p_list
                       if pf <= real_q_on and (real_q_on - po) < int(0.45 * TARGET_FS)]
            if p_cands:
                p_on, p_off = p_cands[-1]
            
            t_on, t_off = None, None
            t_cands = [(to, tf) for to, tf in t_list
                       if to >= real_q_off and (tf - real_q_on) < int(0.9 * TARGET_FS)]
            if t_cands:
                t_on, t_off = t_cands[0]
            
            boundaries = {
                'P_Onset': p_on, 'P_Offset': p_off,
                'QRS_Onset': real_q_on, 'QRS_Offset': real_q_off,
                'T_Onset': t_on, 'T_Offset': t_off,
            }
            
            peaks = find_all_peaks(sig_filt, boundaries, matched, lead_name, fs=TARGET_FS)
            
            for key in ("Q", "R", "S", "T"):
                _mark_peak(axes[2], sig, t_ms, peaks.get(key), key, fs=TARGET_FS, show_text=show_text)
            
            if lead_name.upper() == 'AVR' or lead_name.upper().startswith('V'):
                _mark_peak(axes[2], sig, t_ms, peaks.get("J"), "J", fs=TARGET_FS, show_text=show_text)
            if peaks.get("is_p_bifasica"):
                _mark_peak(axes[2], sig, t_ms, peaks.get("P_prime"), "P", fs=TARGET_FS, show_text=show_text)
            else:
                _mark_peak(axes[2], sig, t_ms, peaks.get("P"), "P", fs=TARGET_FS, show_text=show_text)
            if peaks.get("T_bifasica_secondaria") is not None:
                idx = peaks["T_bifasica_secondaria"]
                axes[2].scatter([t_ms[idx]], [sig[idx]], marker="^", color="#81C784",
                                s=30, zorder=8, edgecolors="white", linewidths=0.4)

    axes[2].set_title(f"3. {model_choice} con Post-Processing", loc='left', color='#555', fontsize=10)
    axes[2].set_xlabel("Tempo (ms)")
    axes[2].legend(loc='upper right', framealpha=1.0)
    axes[2].grid(alpha=0.3)

    return fig

# --- App Streamlit ---
def main():
    models_dict, device = load_models()
    st.sidebar.title("Impostazioni")
    
    st.sidebar.markdown("### Guida al Caricamento")
    with st.sidebar.expander("ℹ️ Cosa caricare e cosa otterrai", expanded=False):
        st.markdown("""
        **Sorgente GE (XML)**
        - **Carica:** Un singolo file `.xml` generato dall'elettrocardiografo (es. GE MAC 2000).
        - **Output:** Visualizzazione per singola derivazione del *Tracciato Completo (10s)* (con Ground Truth ricostruito) e del *Battito Medio (1.2s)* nativo con le relative metriche.
        
        **Sorgente LUDB (WFDB)**
        - **Carica:** Più file contemporaneamente. Obbligatori `.dat` e `.hea`. Per il Ground Truth servono anche i file delle singole annotazioni (es. `.i`, `.v1`, ecc.).
        - **Output:** Visualizzazione per singola derivazione del *Tracciato Completo (10s)* e di un nuovo *Battito Medio Sintetico*, calcolato automaticamente estraendo e mediando i battiti del tracciato continuo.
        """)
        
    data_source = st.sidebar.radio("Sorgente Dati", ["GE (XML)", "LUDB (WFDB)"])
    
    uploaded_files = None
    if data_source == "GE (XML)":
        f = st.sidebar.file_uploader("Carica file GE (.xml)", type=["xml"], accept_multiple_files=False)
        if f: uploaded_files = [f]
    else:
        uploaded_files = st.sidebar.file_uploader("Carica file LUDB (.dat, .hea, annotazioni)", accept_multiple_files=True)
    model_choices = ["Ensemble 2+5+6 (migliore per ST)", "Ensemble 2+5+2 (migliore per solo T)"] + [f"Solo {k}" for k in sorted(list(models_dict.keys()))]
    model_choice = st.sidebar.selectbox("Scelta Modello per Maschere", model_choices)
    
    with st.sidebar.expander("ℹ️ Info Modelli", expanded=False):
        st.markdown("""
**Ensemble 2+5+6 — Segmentazione clinica (separa ST)**
P da Fase 2, QRS da Fase 5, T da Fase 6. Riproduce una vera separazione ST-T (durata mediana ST ≈90ms su LUDB, contro ≈104ms del ground truth manuale). Da usare quando serve accuratezza clinica reale.

**Ensemble 2+5+2 — Compatibilità GE (forma T originale)**
P da Fase 2, QRS da Fase 5, T da Fase 2. Riproduce fedelmente la convenzione GE che non separa lo ST (T_Onset ≈ QRS_Offset, durata ST ≈2ms). Da usare per compatibilità con la convenzione GE originale o come baseline di confronto — non per l'accuratezza clinica.

**Fase 1 — Baseline LUDB**
Addestrata e testata solo su LUDB (annotazione manuale, gold standard). Non ha mai visto dati GE reali. Riferimento per la qualità massima raggiungibile con supervisione perfetta.

**Fase 2 — Baseline PRIV battito medio**
Addestrata sui battiti mediani (600 campioni) estratti dai file GE MAC2000, con l'annotazione automatica GE. In-domain sul formato "beat mediano".

**Fase 3 — Distillazione battito medio**
Come Fase 2, ma con knowledge distillation dal modello LUDB per correggere il bias di annotazione GE (in particolare sulla T). Stesso formato di input di Fase 2.

**Fase 5 — Timbro grezzo (lower bound)**
Addestrata direttamente sul tracciato 10s grezzo GE (nessuna distillazione). Il più robusto sulle onde P e QRS in generalizzazione cross-dominio, ma eredita il bias GE sulla T.

**Fase 6 — Distillazione timbro**
Come Fase 5, ma con distillazione dalla convenzione LUDB per correggere la T (separazione ST). Pensata specificamente per la componente T dell'ensemble.

**Fase 7 — Combined PRIV+LUDB**
Addestrata su un dataset che unisce PRIV timbro e LUDB. Generalizza meglio di Fase 6 sulla T quando testata su segnali GE reali.

**Fase 8 — PRIV timbro in-domain**
Addestrata e testata in-domain sul tracciato 10s GE (nessuna distillazione, come Fase 5 ma verificata specificamente su questo dominio). Riproduce fedelmente la convenzione T originale GE (T_Onset = QRS_Offset, nessun ST).
        """)
    
    if uploaded_files:
        is_ge = (data_source == "GE (XML)")
        if is_ge:
            xml_content = uploaded_files[0].read()
            median_sigs, cont_sigs, global_ann, beat_pos, sample_rate, statements = parse_ge_xml(xml_content)
            ludb_gt_masks = None
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    with open(os.path.join(tmpdir, f.name), "wb") as out:
                        out.write(f.read())
                rec_names = set(f.name.split('.')[0] for f in uploaded_files if f.name.endswith('.hea'))
                if not rec_names:
                    st.error("Carica almeno un file .hea!")
                    return
                rec_name = list(rec_names)[0]
                
                # Estrazione diagnosi dal file .hea
                hea_path = os.path.join(tmpdir, f"{rec_name}.hea")
                with open(hea_path, 'r', encoding='utf-8') as hf:
                    hea_content = hf.read()
                ludb_diag_lines = ludb_parser.parse_hea_diagnoses(hea_content)
                ludb_main_class, ludb_matched, ludb_diag_lines = ludb_parser.get_macro_class_data(ludb_diag_lines)
                
                cont_sigs, ludb_gt_masks = parse_ludb_wfdb(tmpdir, rec_name)
                median_sigs, global_ann, beat_pos, sample_rate = None, None, None, TARGET_FS

        lead_names = list(cont_sigs.keys())
        if not lead_names:
            st.error("Nessuna derivazione trovata.")
            return

        # Controllo se mancano i Ground Truth (se tutte le maschere sono vuote)
        if not is_ge:
            gt_is_empty = all(np.sum(m) == 0 for m in ludb_gt_masks.values())
            if gt_is_empty:
                st.warning("⚠️ **Ground Truth non trovato!** Hai caricato i file `.dat` e `.hea`, ma mancano i file delle annotazioni (es. `.i`, `.ii`, `.v1`, ecc.). Il pannello del Ground Truth rimarrà vuoto. Se vuoi vederlo, ricarica includendo anche quei file!")

        # Tolleranza metriche
        tol_ms = 150
        
        available_leads = list(cont_sigs.keys()) if not is_ge else [l for l in LEADS_ORDER if l in cont_sigs or l in median_sigs]
            
        matched = get_matched_classes(statements) if is_ge else ludb_matched
        
        st.markdown("<h1 style='border-bottom: none; padding-bottom: 0px; margin-bottom: 0px;'>Analisi per derivazione</h1>", unsafe_allow_html=True)

        def run_inference(sig, is_cont=False):
            sig_norm = ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float32)
            if is_cont:
                n_len = len(sig)
                pad_len = (16 - (n_len % 16)) % 16
                sig_pad = np.pad(sig_norm, (0, pad_len), mode='constant')
            else:
                sig_pad = np.pad(sig_norm, (4, 4), mode='constant')
            x = torch.tensor(sig_pad).unsqueeze(0).unsqueeze(0).to(device)
            
            res = {}
            with torch.no_grad():
                if model_choice.startswith("Ensemble 2+5+6"):
                    model_p = models_dict.get("Fase2")
                    model_qrs = models_dict.get("Fase5")
                    model_t = models_dict.get("Fase6")
                    if model_p: res["Fase2_p"] = torch.nn.functional.softmax(model_p(x), dim=1).squeeze(0).cpu()
                    if model_qrs: res["Fase5_qrs"] = torch.nn.functional.softmax(model_qrs(x), dim=1).squeeze(0).cpu()
                    if model_t: res["Fase6_t"] = torch.nn.functional.softmax(model_t(x), dim=1).squeeze(0).cpu()
                elif model_choice.startswith("Ensemble 2+5+2"):
                    model_p = models_dict.get("Fase2")
                    model_qrs = models_dict.get("Fase5")
                    model_t = models_dict.get("Fase2")
                    if model_p: res["Fase2_p"] = torch.nn.functional.softmax(model_p(x), dim=1).squeeze(0).cpu()
                    if model_qrs: res["Fase5_qrs"] = torch.nn.functional.softmax(model_qrs(x), dim=1).squeeze(0).cpu()
                    if model_t: res["Fase2_t"] = torch.nn.functional.softmax(model_t(x), dim=1).squeeze(0).cpu()
                else:
                    target_name = model_choice.replace("Solo ", "")
                    model = models_dict.get(target_name)
                    if model: res[target_name] = torch.nn.functional.softmax(model(x), dim=1).squeeze(0).cpu()
            
            if is_cont:
                return {k: v[:, :n_len] for k, v in res.items()}
            else:
                return {k: v[:, 4:604] for k, v in res.items()}

        def get_mask(scores_dict, length):
            mask = np.zeros(length, dtype=np.int64)
            if model_choice.startswith("Ensemble 2+5+6"):
                s_p = scores_dict.get("Fase2_p")
                s_qrs = scores_dict.get("Fase5_qrs")
                s_t = scores_dict.get("Fase6_t")
                p_p = s_p.argmax(0).numpy() if s_p is not None else np.zeros(length)
                p_qrs = s_qrs.argmax(0).numpy() if s_qrs is not None else np.zeros(length)
                p_t = s_t.argmax(0).numpy() if s_t is not None else np.zeros(length)
                mask[p_t == 3] = 3
                mask[p_p == 1] = 1
                mask[p_qrs == 2] = 2
            elif model_choice.startswith("Ensemble 2+5+2"):
                s_p = scores_dict.get("Fase2_p")
                s_qrs = scores_dict.get("Fase5_qrs")
                s_t = scores_dict.get("Fase2_t")
                p_p = s_p.argmax(0).numpy() if s_p is not None else np.zeros(length)
                p_qrs = s_qrs.argmax(0).numpy() if s_qrs is not None else np.zeros(length)
                p_t = s_t.argmax(0).numpy() if s_t is not None else np.zeros(length)
                mask[p_t == 3] = 3
                mask[p_p == 1] = 1
                mask[p_qrs == 2] = 2
                
                # Remove ST gap by extending QRS wave to touch T onset
                last_2 = -1
                for i in range(length):
                    if mask[i] == 2: last_2 = i
                    elif mask[i] == 3:
                        if last_2 != -1: mask[last_2+1:i] = 2
                        last_2 = -1
                    elif mask[i] == 1: last_2 = -1
            else:
                target_name = model_choice.replace("Solo ", "")
                s_model = scores_dict.get(target_name)
                if s_model is not None:
                    pm = s_model.argmax(0).numpy()
                    mask[pm == 2] = 2
                    mask[pm == 1] = 1
                    mask[pm == 3] = 3
            return mask

        def render_metrics_tables(gt_mask, masks_dict, current_tol, container=st):
            import pandas as pd
            if np.sum(gt_mask) == 0:
                container.info("Ground Truth vuoto. Metriche non disponibili.")
                return
            res = {}
            for name, mask in masks_dict.items():
                res[name] = compute_metrics_for_app(mask, gt_mask, tol_ms=current_tol)

            def make_df(c_idx, c_name):
                data = []
                for name in masks_dict.keys():
                    m = res[name]
                    f1 = m[f'f1_{c_name}']
                    se = m[f'se_{c_name}']
                    ppv = m[f'ppv_{c_name}']
                    data.append([name, f"{se:.2f}%", f"{ppv:.2f}%", f"{f1:.2f}%"])
                return pd.DataFrame(data, columns=["Metodo / Fase", "Sensitivity", "Precision (PPV)", "F1-score"])

            container.markdown("##### 🔵 Metriche Onda P")
            container.table(make_df('P', 'p'))
            container.markdown("##### 🔴 Metriche Complesso QRS")
            container.table(make_df('QRS', 'q'))
            container.markdown("##### 🟢 Metriche Onda T")
            container.table(make_df('T', 't'))

        
        global_ann_corrected = global_ann.copy() if global_ann else {}
        if is_ge:
            global_ann_corrected = get_corrected_pacing_ann(global_ann, median_sigs, matched)

        col_main, col_side = st.columns([3, 1])
        with col_side:

            def traduci_statement(testo):
                # Ordiniamo dalla frase più lunga alla più corta per evitare sostituzioni parziali errate
                mappa = {
                    "Non-specific repolarization abnormalities": "Anomalie aspecifiche della ripolarizzazione",
                    "Left bundle branch block": "Blocco di branca sinistra (BBS)",
                    "Right bundle branch block": "Blocco di branca destra (BBD)",
                    "Incomplete right bundle branch block": "Blocco di branca destra incompleto",
                    "Complete heart block": "Blocco AV completo (III grado)",
                    "First degree AV block": "Blocco AV di I grado",
                    "Second degree AV block": "Blocco AV di II grado",
                    "Premature ventricular contraction": "Battito ectopico ventricolare (BEV)",
                    "Premature atrial contraction": "Battito ectopico atriale (BEA)",
                    "Left ventricular hypertrophy": "Ipertrofia ventricolare sinistra",
                    "Right ventricular hypertrophy": "Ipertrofia ventricolare destra",
                    "Anterior myocardial infarction": "Infarto miocardico anteriore",
                    "Inferior myocardial infarction": "Infarto miocardico inferiore",
                    "Left ventricular overload": "Sovraccarico ventricolare sinistro",
                    "Right ventricular overload": "Sovraccarico ventricolare destro",
                    "Wolff-Parkinson-White": "Sindrome di Wolff-Parkinson-White",
                    "ST segment depression": "Sottoslivellamento del tratto ST",
                    "ST segment elevation": "Sovraslivellamento del tratto ST",
                    "Left axis deviation": "Deviazione assiale sinistra",
                    "Right axis deviation": "Deviazione assiale destra",
                    "Normal sinus rhythm": "Ritmo sinusale normale",
                    "Myocardial infarction": "Infarto miocardico",
                    "Ventricular pacing": "Pacing ventricolare",
                    "Atrial fibrillation": "Fibrillazione atriale",
                    "T wave abnormality": "Anomalia dell'onda T",
                    "T wave inversion": "Inversione dell'onda T",
                    "Sinus bradycardia": "Bradicardia sinusale",
                    "Sinus tachycardia": "Tachicardia sinusale",
                    "Atrial flutter": "Flutter atriale",
                    "Atrial pacing": "Pacing atriale",
                    "Sinus rhythm": "Ritmo sinusale",
                    "Normal ECG": "ECG nei limiti della norma",
                    "posterior wall": "parete posteriore",
                    "anterior wall": "parete anteriore",
                    "inferior wall": "parete inferiore",
                    "lateral wall": "parete laterale",
                    "anteroseptal": "anterosettale",
                    "Ischemia": "Ischemia",
                    "Pacing": "Ritmo da Pacemaker",
                }
                
                risultato = testo
                # Eseguiamo le sostituzioni pezzo per pezzo (case-insensitive in ricerca, ma manteniamo il case della traduzione)
                import re
                for eng, ita in mappa.items():
                    # Sostituzione case-insensitive usando regex
                    risultato = re.sub(re.escape(eng), ita, risultato, flags=re.IGNORECASE)
                
                return risultato

            st.markdown("<h5 style='margin-top: 60px; margin-bottom: 10px;'>Selezione Derivazione</h5>", unsafe_allow_html=True)
            selected_lead = st.selectbox("Scegli la derivazione", available_leads, index=0, label_visibility="collapsed")
            selected_leads = [selected_lead] if selected_lead else []
            
            show_peaks = st.checkbox("Visualizza Picchi (P, Q, R, S, T, J) sul battito mediano", value=True)
            st.caption("i picchi riportati per questo file sono ottenuti dal modello Ensemble senza confronto con annotazioni manuali di riferimento — trattarli come stime, non come misure validate")
            
            if not selected_leads:
                st.warning("Nessuna derivazione disponibile.")
                return
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if not is_ge:
                st.markdown("##### Diagnosi Paziente")
                st.markdown(f"**Macro-Classe:** `{ludb_main_class}`")
                if ludb_matched:
                    st.markdown(f"**Classi attivate:** {', '.join(ludb_matched)}")
                
                with st.expander("Diagnosi Originali", expanded=True):
                    if ludb_diag_lines:
                        for d in ludb_diag_lines:
                            st.write(f"- {traduci_statement(d)}")
                    else:
                        st.write("Nessuna diagnosi trovata")
            else:
                st.markdown("##### Diagnosi Paziente")
                if matched:
                    st.markdown(f"**Classi GE attivate:** {', '.join(matched)}")
                else:
                    st.markdown(f"**Classi GE attivate:** Standard")
                    
                with st.expander("Statement Originali", expanded=True):
                    if statements:
                        grouped = []
                        for s in statements:
                            t = traduci_statement(s)
                            if not t: continue
                            # Rimuove asterischi decorativi (es. "** MI ACUTO **")
                            t = t.replace("*", "").strip()
                            if not t: continue
                            if not grouped:
                                grouped.append(t)
                            else:
                                # Unisci al precedente se inizia con virgola, punto e virgola, minuscola o congiunzione
                                if t[0] in (',', ';') or t[0].islower() or t.lower().startswith("con ") or t.lower().startswith("ed ") or t.lower().startswith("e "):
                                    grouped[-1] += " " + t.lstrip(",; ")
                                else:
                                    grouped.append(t)
                        for g in grouped:
                            st.write(f"- {g}")
                    else:
                        st.write("Nessuna diagnosi trovata")
        
        if not is_ge:
            
            # Pre-calcolo inferenza per LUDB (che usa tutto l'elenco)
            lead_scores = {}
            for l, s_ in cont_sigs.items():
                lead_scores[l] = run_inference(s_, is_cont=True)

            ludb_median_sigs = {}

            for cont_lead in selected_leads:
                cont_raw = cont_sigs.get(cont_lead)
                if cont_raw is None: continue
                n_len = len(cont_raw)
                
                scores_single = lead_scores[cont_lead]
                mask_single_pre = get_mask(scores_single, n_len)
                mask_single_post = postprocess_mask_multibeat(mask_single_pre, fs=TARGET_FS)
                
                pred_mask = mask_single_pre
                pred_post = mask_single_post
                gt_mask = ludb_gt_masks[cont_lead]
                
                max_samples = min(n_len, 5000)
                sig_plot = cont_raw[:max_samples]
                gt_mask_plot = gt_mask[:max_samples]
                pred_mask_plot = pred_mask[:max_samples]
                pred_post_plot = pred_post[:max_samples]
                
                col_main.markdown(f"<h4 style='margin-top: 0px;'>DERIVAZIONE {cont_lead} - tracciato 10s + battito mediano</h4>", unsafe_allow_html=True)
                
                if np.std(cont_raw) < 1e-4 or np.max(cont_raw) == np.min(cont_raw):
                    col_main.error("⚠️ **ATTENZIONE:** Il segnale per questa derivazione risulta piatto o assente nel file caricato. Il tracciato e il battito medio appariranno come una linea retta e le predizioni non saranno attendibili per questa specifica derivazione.")
                
                
                fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                         f"Derivazione {cont_lead}", cont_lead, False, matched, show_text=False, model_choice=model_choice, show_gt_st=True)
                col_main.pyplot(fig_cont, use_container_width=True)
                
                # Calcola il battito medio E la maschera GT mediana
                med_beat, med_gt_mask = compute_median_beat_from_continuous(cont_raw, mask_single_post, gt_mask=gt_mask, fs=TARGET_FS, median_len=MEDIAN_LEN)
                
                if med_beat is not None:
                    # Mostra subito il battito medio sotto al tracciato di questa derivazione
                    col_main.markdown(f"**Battito Medio Sintetico - Derivazione {cont_lead}**")
                    fig_med = plot_single_median_beat(cont_lead, med_beat, med_gt_mask, (models_dict, device), matched, fs=TARGET_FS, show_peaks=show_peaks, model_choice=model_choice, show_gt_st=True)
                    col_main.pyplot(fig_med, use_container_width=True)
                
        else:
            
            gt_mask_base = build_median_gt_mask(global_ann_corrected, TARGET_FS)
            
            for lead in selected_leads:
                if lead not in cont_sigs and lead not in median_sigs:
                    continue
                    
                col_main.markdown(f"<h4 style='margin-top: 0px;'>DERIVAZIONE {lead} - tracciato 10s + battito mediano</h4>", unsafe_allow_html=True)
                
                sig_check = cont_sigs[lead] if lead in cont_sigs else median_sigs[lead]
                if np.std(sig_check) < 1e-4 or np.max(sig_check) == np.min(sig_check):
                    col_main.error("⚠️ **ATTENZIONE:** Il segnale per questa derivazione risulta piatto o assente nel file caricato. Il tracciato e il battito medio appariranno come una linea retta e le predizioni non saranno attendibili per questa specifica derivazione.")
                
                
                # 1. Tracciato Continuo (se disponibile per questa derivazione)
                if lead in cont_sigs:
                    cont_raw = cont_sigs[lead]
                    n_len = len(cont_raw)
                    sig_med_cont = median_sigs.get(lead)
                    if sig_med_cont is not None:
                        scores_dict = run_inference(sig_med_cont, is_cont=False)
                        mask_med_pre = get_mask(scores_dict, MEDIAN_LEN)
                        mask_med_post = postprocess_mask(mask_med_pre, fs=TARGET_FS)
                    else:
                        mask_med_pre = np.zeros(MEDIAN_LEN, dtype=np.int64)
                        mask_med_post = np.zeros(MEDIAN_LEN, dtype=np.int64)
                    
                    pred_mask = stamp_median_mask(beat_pos, sample_rate, mask_med_pre, n_len)
                    pred_post = stamp_median_mask(beat_pos, sample_rate, mask_med_post, n_len)
                    gt_mask = build_continuous_mask_from_ann(beat_pos, sample_rate, global_ann_corrected, n_len)
                    
                    max_samples = min(n_len, 10000)
                    sig_plot = cont_raw[:max_samples]
                    gt_mask_plot = gt_mask[:max_samples]
                    pred_mask_plot = pred_mask[:max_samples]
                    pred_post_plot = pred_post[:max_samples]
                    
                    if 'PACING' in matched:
                        col_main.warning("⚠ Attenzione: per la classe PACING il ground truth su questo tracciato a 10s è meno affidabile. Le bande derivano dal beat mediano GE, che cattura solo la morfologia dominante del paziente — se il tracciato mescola battiti intrinseci e stimolati, le annotazioni possono non essere corrette su alcuni battiti del tracciato.")
                    
                    fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                             f"Derivazione {lead}", lead, False, matched, show_text=False, model_choice=model_choice)
                    col_main.pyplot(fig_cont, use_container_width=True)
                    
                # 2. Battito Mediano (se disponibile)
                if lead in median_sigs:
                    sig = median_sigs[lead]
                    fig_med = plot_single_median_beat(lead, sig, gt_mask_base, (models_dict, device), matched, fs=TARGET_FS, show_peaks=show_peaks, model_choice=model_choice)
                    col_main.pyplot(fig_med, use_container_width=True)
                    

            # Metriche globali battito medio (COMMENTATE)
            # if median_sigs:
            #     col_main.markdown(f"#### Metriche sul Battito Medio (Tolleranza {tol_ms}ms) - Media Ensemble")
            #     lead_scores_v3, lead_scores_f6 = {}, {}
            #     for l, sig in median_sigs.items():
            #         s3, s6 = run_inference(sig, is_cont=False)
            #         lead_scores_v3[l] = s3
            #         lead_scores_f6[l] = s6
            #         
            #     sv3_avg = torch.stack(list(lead_scores_v3.values()), dim=0).mean(dim=0)
            #     sf6_avg = torch.stack(list(lead_scores_f6.values()), dim=0).mean(dim=0)
            #     mask_avg_pre = get_mask(sv3_avg, sf6_avg, MEDIAN_LEN)
            #     mask_avg_post = postprocess_mask(mask_avg_pre, fs=TARGET_FS)
            #     
            #     gt_med_mask = build_median_gt_mask(global_ann_corrected, TARGET_FS)
            #     masks_dict = {
            #         "Avg12 - Pre-PP (Fase 2)": mask_avg_pre,
            #         "Avg12 - Post-PP (Fase 6)": mask_avg_post,
            #     }
            #     
            #     # Cerca derivazione II per metriche specifiche se disponibile
            #     lead_ii_key = next((k for k in median_sigs.keys() if k.lower() == 'ii'), None)
            #     if lead_ii_key:
            #         mask_ii_pre = get_mask(lead_scores_v3[lead_ii_key], lead_scores_f6[lead_ii_key], MEDIAN_LEN)
            #         masks_dict["Lead II - Post-PP"] = postprocess_mask(mask_ii_pre, fs=TARGET_FS)
            #         
            #     render_metrics_tables(gt_med_mask, masks_dict, tol_ms, container=col_main)

if __name__ == "__main__":


    main()
