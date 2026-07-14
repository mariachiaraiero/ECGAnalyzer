"""
Test di segmentazione ECG su file XML (formato GE Mac2000).
Ensemble Fase2 V3 (P e QRS) + Fase6 (T e ST).

Produco un CSV con le metriche cliniche per ogni file analizzato.

# uso : python test_ensemble_fasi.py --xml-dir percorso_cartella_filexml

"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#  architettura: Enhanced U-Net V3 (da Fase2 — compatibile con best_fase2_v3.pt)

class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//r), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(x.mean(-1)).view(b, c, 1)

class AttBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 9, padding=4), nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, 9, padding=4), nn.BatchNorm1d(out_ch),
            SEBlock(out_ch)
        )
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
                         if in_ch != out_ch else nn.Identity())
    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x), inplace=True)

class MultiScaleInput(nn.Module):
    def __init__(self, in_ch=1, out_ch=12):
        super().__init__()
        c = out_ch // 3
        self.s1 = nn.Sequential(nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(nn.AvgPool1d(2), nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(nn.AvgPool1d(4), nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
    def forward(self, x):
        L = x.size(2)
        return torch.cat([self.s1(x),
                          F.interpolate(self.s2(x), size=L),
                          F.interpolate(self.s3(x), size=L)], 1)

class EnhancedUNetV3(nn.Module):
    """Enhanced U-Net V3 — architettura Fase2 (compatibile con best_fase2_v3.pt)."""
    def __init__(self, in_channels=1, num_classes=4, f=12, dropout=0.15):
        super().__init__()
        self.ms = MultiScaleInput(in_channels, f)
        self.enc1 = AttBlock(f,    f,    dropout)
        self.enc2 = AttBlock(f,    f*2,  dropout)
        self.enc3 = AttBlock(f*2,  f*4,  dropout)
        self.enc4 = AttBlock(f*4,  f*8,  dropout)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = AttBlock(f*8, f*16, dropout)
        self.up   = nn.ModuleList([nn.ConvTranspose1d(f*i, f*i, 8, 2, 3)
                                   for i in [16, 8, 4, 2]])
        self.dec  = nn.ModuleList([AttBlock(f*24, f*8,  dropout),
                                   AttBlock(f*12, f*4,  dropout),
                                   AttBlock(f*6,  f*2,  dropout),
                                   AttBlock(f*3,  f,    dropout)])
        self.final = nn.Conv1d(f, num_classes, 1)
        self.ds4   = nn.Conv1d(f*8, num_classes, 1)
        self.ds3   = nn.Conv1d(f*4, num_classes, 1)

    def forward(self, x):
        L  = x.size(2)
        ms = self.ms(x)
        e1 = self.enc1(ms)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        def pm(xu, xs): return F.interpolate(xu, size=xs.size(2))
        d4 = self.dec[0](torch.cat([pm(self.up[0](b),  e4), e4], 1))
        d3 = self.dec[1](torch.cat([pm(self.up[1](d4), e3), e3], 1))
        d2 = self.dec[2](torch.cat([pm(self.up[2](d3), e2), e2], 1))
        d1 = self.dec[3](torch.cat([pm(self.up[3](d2), e1), e1], 1))
        return self.final(d1)




class MultiScaleInputFase6(nn.Module):
    def __init__(self, in_ch=1, out_ch=12):
        super().__init__()
        c = out_ch // 3
        self.s1 = nn.Sequential(nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(nn.AvgPool1d(kernel_size=2, stride=2), nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(nn.AvgPool1d(kernel_size=4, stride=4), nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
    def forward(self, x):
        L = x.size(2)
        return torch.cat([self.s1(x), F.interpolate(self.s2(x), size=L, mode='linear', align_corners=False), F.interpolate(self.s3(x), size=L, mode='linear', align_corners=False)], dim=1)

class ECGUNetFase6(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_filters=12, dropout=0.15):
        super().__init__()
        f = base_filters
        self.ms = MultiScaleInputFase6(in_channels, f)
        self.enc1 = AttBlock(f, f, dropout=dropout)
        self.enc2 = AttBlock(f, f*2, dropout=dropout)
        self.enc3 = AttBlock(f*2, f*4, dropout=dropout)
        self.enc4 = AttBlock(f*4, f*8, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = AttBlock(f*8, f*16, dropout=dropout)
        self.up = nn.ModuleList([
            nn.ConvTranspose1d(f*16, f*16, kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f*8,  f*8,  kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f*4,  f*4,  kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f*2,  f*2,  kernel_size=8, stride=2, padding=3),
        ])
        self.dec = nn.ModuleList([
            AttBlock(f*24, f*8,  dropout=dropout),
            AttBlock(f*12, f*4,  dropout=dropout),
            AttBlock(f*6,  f*2,  dropout=dropout),
            AttBlock(f*3,  f,    dropout=dropout),
        ])
        self.final = nn.Conv1d(f, num_classes, kernel_size=1)
        self.ds4   = nn.Conv1d(f*8, num_classes, kernel_size=1)
        self.ds3   = nn.Conv1d(f*4, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x, n):
        d = n - x.size(2)
        if d > 0:  return F.pad(x, (d // 2, d - d // 2))
        if d < 0:  return x[:, :, :n]
        return x

    def forward(self, x):
        ms = self.ms(x)
        e1 = self.enc1(ms)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        skips = [e4, e3, e2, e1]
        d = b
        for i, (up, dec_block, skip) in enumerate(zip(self.up, self.dec, skips)):
            d = up(d)
            d = self._pad_to_match(d, skip.size(2))
            d = torch.cat([d, skip], dim=1)
            d = dec_block(d)
        return self.final(d)

#cost
LEADS_ORDER = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
CLASS_NAMES = {0: 'Background', 1: 'Onda P', 2: 'QRS', 3: 'Onda T'}
MEDIAN_LEN  = 600
TARGET_FS   = 500
NUM_CLASSES = 4

C_P   = '#2196F3'
C_QRS = '#E53935'
C_ST  = '#FFC107'
C_T   = '#43A047'
C_TGT = '#FF7043'

# Macro-classi diagnostiche 
MACRO_CLASS_NAMES = {
    0: 'HEALTHY', 1: 'PACING', 2: 'NO_P', 3: 'WIDE_QRS',
    4: 'OLD_MI', 5: 'AV_BLOCK', 6: 'ST_T', 7: 'HYPERTROPHY', 8: 'UNKNOWN',
}
PRIORITY = ['PACING','NO_P','WIDE_QRS','OLD_MI','AV_BLOCK','ST_T','HYPERTROPHY','HEALTHY']
CLASS_MAPPING = {
    'PACING':      ['pacing atriale','pacing ventricolare','stimolazione','pacemaker'],
    'NO_P':        ['fibrillazione atriale','flutter atriale','ritmo giunzionale',
                    'ritmo idioventricolare','centro giunzionale'],
    'WIDE_QRS':    ['blocco di branca','emiblocco','allargamento del qrs',
                    'allargamento qrs','conduzione intraventricolare',
                    'blocco intraventricolare','ritardo della conduzione'],
    'OLD_MI':      ['infarto','onda q patolog'],
    'AV_BLOCK':    ['blocco a-v','bav','atrio-ventricolare','atrioventricolare',
                    'p-r corto','pr corto'],
    'ST_T':        ['st aspecifiche','anormalit','ischemia','depressione st',
                    'sottoslivellamento','sopraslivellamento','ripolarizzazione',
                    'sovraccarico','pericardite','miocardite','qt lungo',
                    'allungamento del qt'],
    'HYPERTROPHY': ['ipertrofia','ingrandimento atriale','ivs',
                    'deviazione assiale sinistra','deviazione assiale destra',
                    'basso voltaggio'],
    'HEALTHY':     ['ecg normale','ritmo sinusale normale','bradicardia sinusale',
                    'tachicardia sinusale','aritmia sinusale','peraltro ecg normale'],
}
ARTIFACT_KEYWORDS = [
    'tremore muscolare','interferenza linea','interferenza della linea',
    'rumore elettrodi','qualita dati scadente','baseline wander','disturbo linea base',
]

def get_macro_class(statements):
    """Assegna macro-classe diagnostica dalla lista di statement GE."""
    txt      = ' '.join(statements).lower() if statements else ''
    matched  = set()
    has_art  = int(any(kw in txt for kw in ARTIFACT_KEYWORDS))
    for mc, kws in CLASS_MAPPING.items():
        if any(kw in txt for kw in kws):
            matched.add(mc)
    main_cls = 'UNKNOWN'
    for p in PRIORITY:
        if p in matched:
            main_cls = p; break
    if 'ecg anormale' in txt:       final_j = 'abnormal'
    elif 'limiti della norma' in txt or 'borderline' in txt: final_j = 'borderline'
    elif 'ecg normale' in txt or 'peraltro ecg normale' in txt: final_j = 'normal'
    else:                           final_j = 'unknown'
    return (main_cls, has_art,
            int(final_j != 'normal'),   # is_path_strict
            int(final_j == 'abnormal')) # is_path_lax


#parsing xml GE

def parse_xml(xml_path):
    """
    Estrae dal file XML GE Mac2000:
      - signals      : dict {lead: np.array(600,) in mV}
      - global_ann   : dict con annotazioni temporali in ms
      - statements   : lista statement diagnostici per macro-classe
      - per_lead_ann : dict {lead: {tag: valore in uV}} per le ampiezze
    Ritorna None se il file non è leggibile o non contiene median beat.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    ns = {"ns": "urn:ge:sapphire:dcar_1"}
    all_wf = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)
    if not all_wf:
        ns = {"ns": "urn:ge:sapphire:sapphire_3"}
        all_wf = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)
    if not all_wf:
        return None

    median_wf = [
        wf for wf in all_wf
        if wf.get('asizeVT', '0') == str(MEDIAN_LEN)
        or wf.get('asizeBT', '0') == str(MEDIAN_LEN)
    ]
    if not median_wf:
        return None

    signals = {}
    for wf in median_wf:
        lead    = wf.get('lead', '').upper()
        vt_data = wf.get('V', '')
        if not lead or not vt_data:
            continue
        try:
            raw_vals = [int(v) for v in vt_data.split()]
        except ValueError:
            continue

        clean_vals, last_valid = [], 0
        for v in raw_vals:
            if v != -32768:
                clean_vals.append(v)
                last_valid = v
            else:
                clean_vals.append(last_valid)

        if len(clean_vals) > MEDIAN_LEN:
            clean_vals = clean_vals[:MEDIAN_LEN]
        elif len(clean_vals) < MEDIAN_LEN:
            clean_vals += [last_valid] * (MEDIAN_LEN - len(clean_vals))

        signals[lead] = np.array(
            [4.88 * v / 1000.0 for v in clean_vals], dtype=np.float32
        )

    if not signals:
        return None

    global_ann = {}
    g_node = root.find(
        ".//ns:medianTemplate/ns:measurements/ns:global", namespaces=ns
    )
    if g_node is not None:
        for child in g_node:
            v = child.get('V')
            if v and v != '-32768':
                tag = child.tag.split('}')[-1]
                try:
                    global_ann[tag] = int(v)
                except ValueError:
                    pass

    # Statement diagnostici
    statements = []
    interp_node = root.find(".//ns:interpretation", namespaces=ns)
    if interp_node is not None:
        for s in interp_node.findall(".//ns:statement", namespaces=ns):
            val = s.get('V')
            if val and val.strip():
                statements.append(val.strip())

    # Annotazioni per singola derivazione (Ampiezze in uV, ecc.)
    per_lead_ann = {}
    l_nodes = root.findall(".//ns:medianTemplate/ns:measurements/ns:perLead", namespaces=ns)
    for l_node in l_nodes:
        lname = l_node.get('lead', '').upper()
        if not lname: continue
        per_lead_ann[lname] = {}
        for child in l_node:
            v = child.get('V')
            if v and v != '-32768':
                tag = child.tag.split('}')[-1]
                try:
                    per_lead_ann[lname][tag] = int(v)
                except ValueError:
                    pass

    return signals, global_ann, statements, per_lead_ann


#maschera

def build_mask(ann):
    """Costruisce la maschera [600] dalle annotazioni globali del medianTemplate."""
    mask = np.zeros(MEDIAN_LEN, dtype=np.int64)

    def ms2s(ms):
        return max(0, min(int(ms * TARGET_FS / 1000), MEDIAN_LEN - 1))

    p_on  = ann.get('P_Onset');   p_off = ann.get('P_Offset')
    q_on  = ann.get('Q_Onset');   q_off = ann.get('Q_Offset')
    t_off = ann.get('T_Offset')
    t_on  = ann.get('T_Onset')
    if t_on is None:
        t_on = ann.get('Q_Offset')

    if p_on is not None and p_off is not None:
        s, e = ms2s(p_on), ms2s(p_off)
        if s < e: mask[s:e+1] = 1
    if q_on is not None and q_off is not None:
        s, e = ms2s(q_on), ms2s(q_off)
        if s < e: mask[s:e+1] = 2
    if t_on is not None and t_off is not None and t_off > t_on:
        s, e = ms2s(t_on), ms2s(t_off)
        if s < e: mask[s:e+1] = 3

    return mask


#normalizz

def normalize(sig):
    """Normalizzazione Z-score per campione."""
    std = np.std(sig)
    if std < 1e-6:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


def apply_ecg_filters(signal, fs=500):
    """Applica filtri Notch (50Hz) e High-Pass (0.5Hz) come nel training."""
    if np.all(signal == 0):
        return signal
    try:
        # Notch 50Hz
        Q = 30.0
        w0 = 50.0 / (fs / 2)
        b_notch, a_notch = iirnotch(w0, Q)
        sig_notch = filtfilt(b_notch, a_notch, signal)
        
        # High-Pass 0.5Hz
        cutoff = 0.5 / (fs / 2)
        b_hp, a_hp = butter(4, cutoff, btype='high')
        return filtfilt(b_hp, a_hp, sig_notch)
    except Exception:
        return signal


#metriche cliniche

def extract_intervals(mask, fs=500):
    """Estrae onset/offset (in ms) dalla maschera di segmentazione."""
    def get_pts(m, cid):
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on  = np.where(diff ==  1)[0]
        off = np.where(diff == -1)[0] - 1
        return (on[0] if len(on) > 0 else None,
                off[0] if len(off) > 0 else None)

    p_on, p_off = get_pts(mask, 1)
    q_on, q_off = get_pts(mask, 2)
    t_on, t_off = get_pts(mask, 3)

    res = {
        'P_Onset':     p_on  * 1000/fs if p_on  is not None else None,
        'P_Offset':    p_off * 1000/fs if p_off is not None else None,
        'QRS_Onset':   q_on  * 1000/fs if q_on  is not None else None,
        'QRS_Offset':  q_off * 1000/fs if q_off is not None else None,
        'T_Onset':     t_on  * 1000/fs if t_on  is not None else None,
        'T_Offset':    t_off * 1000/fs if t_off is not None else None,
    }

    if res['QRS_Onset'] is not None and res['P_Onset'] is not None:
        res['PR_Interval'] = res['QRS_Onset'] - res['P_Onset']
    else:
        res['PR_Interval'] = None

    if res['T_Offset'] is not None and res['QRS_Onset'] is not None:
        res['QT_Interval'] = res['T_Offset'] - res['QRS_Onset']
    else:
        res['QT_Interval'] = None

    if res['P_Offset'] is not None and res['P_Onset'] is not None:
        res['P_Duration'] = res['P_Offset'] - res['P_Onset']
    else:
        res['P_Duration'] = None

    if res['QRS_Offset'] is not None and res['QRS_Onset'] is not None:
        res['QRS_Duration'] = res['QRS_Offset'] - res['QRS_Onset']
    else:
        res['QRS_Duration'] = None

    return res

def extract_amplitudes(sig, mask, fs=500):
    """
    Estrae le ampiezze dei picchi (y in mV) e le loro posizioni (x in ms) 
    usando la logica GE decriptata: Baseline puntuale all'Onset.
    """
    res = {}
    
    # Inizializzazione
    for w in ['P', 'Q', 'R', 'S', 'J', 'T']:
        res[f'{w}_Amp'] = None
        res[f'{w}_Peak_ms'] = None

    # Indici delle classi
    p_idx = np.where(mask == 1)[0]
    qrs_idx = np.where(mask == 2)[0]
    t_idx = np.where(mask == 3)[0]

    # 1. Complesso QRS (Punto di riferimento per Q, R, S, J)
    if len(qrs_idx) > 0:
        idx_on_qrs = qrs_idx[0]
        idx_off_qrs = qrs_idx[-1]
        baseline_qrs = sig[idx_on_qrs]  # Baseline REVERSE ENGINEERED
        
        qrs_sig = sig[idx_on_qrs : idx_off_qrs + 1]
        
        # Picco R: massimo assoluto nel recinto
        r_rel_idx = np.argmax(qrs_sig)
        res['R_Amp'] = round(float(qrs_sig[r_rel_idx] - baseline_qrs), 3)
        res['R_Peak_ms'] = round(float((idx_on_qrs + r_rel_idx) * 1000 / fs), 1)
        
        # Onda Q: minimo tra inizio e picco R
        if r_rel_idx > 0:
            q_part = qrs_sig[:r_rel_idx]
            q_rel_idx = np.argmin(q_part)
            res['Q_Amp'] = round(float(abs(q_part[q_rel_idx] - baseline_qrs)), 3)
            res['Q_Peak_ms'] = round(float((idx_on_qrs + q_rel_idx) * 1000 / fs), 1)
        else:
            res['Q_Amp'] = 0.0
            
        # Onda S: minimo tra picco R e fine QRS
        if r_rel_idx < len(qrs_sig) - 1:
            s_part = qrs_sig[r_rel_idx+1:]
            s_rel_idx = np.argmin(s_part)
            res['S_Amp'] = round(float(abs(s_part[s_rel_idx] - baseline_qrs)), 3)
            res['S_Peak_ms'] = round(float((idx_on_qrs + r_rel_idx + 1 + s_rel_idx) * 1000 / fs), 1)
        else:
            res['S_Amp'] = 0.0
            
        # Punto J: valore puntuale all'Offset del QRS
        res['J_Amp'] = round(float(sig[idx_off_qrs] - baseline_qrs), 3)
        res['J_Peak_ms'] = round(float(idx_off_qrs * 1000 / fs), 1)

        # 2. Onda T (Riferita alla fine del QRS se T_Onset manca)
        if len(t_idx) > 0:
            baseline_t = sig[idx_off_qrs] # GE usa Q_Offset come baseline per la T
            t_sig = sig[t_idx[0] : t_idx[-1] + 1]
            # Cerchiamo la deflessione massima (positiva o negativa)
            t_diff = t_sig - baseline_t
            rel_peak_idx = np.argmax(np.abs(t_diff))
            res['T_Amp'] = round(float(t_diff[rel_peak_idx]), 3)
            res['T_Peak_ms'] = round(float((t_idx[0] + rel_peak_idx) * 1000 / fs), 1)

    # 3. Onda P (Indipendente, riferita al suo Onset)
    if len(p_idx) > 0:
        idx_on_p = p_idx[0]
        baseline_p = sig[idx_on_p]
        p_sig = sig[p_idx[0] : p_idx[-1] + 1]
        p_diff = p_sig - baseline_p
        rel_peak_idx = np.argmax(np.abs(p_diff))
        res['P_Amp'] = round(float(p_diff[rel_peak_idx]), 3)
        res['P_Peak_ms'] = round(float((idx_on_p + rel_peak_idx) * 1000 / fs), 1)
            
    return res



def compute_sample_f1(pred, target, num_classes=4):
    """F1 per classe e macro per un singolo campione."""
    res, f1s = {}, []
    for c in range(1, num_classes):
        p = (pred == c).astype(float)
        t = (target == c).astype(float)
        tp = (p * t).sum()
        f1  = (2*tp + 1e-8) / (p.sum() + t.sum() + 1e-8) * 100
        se  = tp / (t.sum() + 1e-8) * 100
        ppv = tp / (p.sum() + 1e-8) * 100
        res[f'f1_c{c}']  = f1
        res[f'se_c{c}']  = se
        res[f'ppv_c{c}'] = ppv
        f1s.append(f1 / 100.0)
    res['f1_macro']  = np.mean(f1s) * 100
    res['accuracy']  = (pred == target).mean() * 100
    return res


def compute_timing_errors(pred, target, fs=500):
    """Calcola gli errori in ms tra predizione e ground truth."""
    p_int = extract_intervals(pred, fs)
    t_int = extract_intervals(target, fs)
    errors = {}
    for key in ['P_Onset','P_Offset','QRS_Onset','QRS_Offset','T_Onset','T_Offset']:
        if p_int[key] is not None and t_int[key] is not None:
            errors[f'err_{key}_ms'] = p_int[key] - t_int[key]
        else:
            errors[f'err_{key}_ms'] = None
    for key in ['PR_Interval','QT_Interval','P_Duration','QRS_Duration']:
        if p_int[key] is not None and t_int[key] is not None:
            errors[f'err_{key}_ms'] = p_int[key] - t_int[key]
        else:
            errors[f'err_{key}_ms'] = None
    return errors


def get_clinical_judgment(error_ms):
    abs_err = abs(error_ms)
    if abs_err <= 5:  return "Eccellente"
    if abs_err <= 15: return "Molto Buono"
    if abs_err <= 30: return "Buono"
    if abs_err <= 50: return "Accettabile"
    return "Fuori Tolleranza"

def shade(ax, a, b, color, alpha, label=None):
    if a is not None and b is not None and b > a:
        ax.axvspan(a, b, alpha=alpha, color=color, label=label, zorder=2)

def vline(ax, x, color='#9E9E9E', ls=':'):
    if x is not None:
        ax.axvline(x, color=color, linewidth=0.7, linestyle=ls, zorder=4, alpha=0.8)


#logica analisi e plot

def plot_segmentation(sig, gt_mask, pred_mask, fname, output_path,
                      fs=TARGET_FS, lead='II'):
    t_ms     = np.arange(len(sig)) * 1000 / fs
    gt_ivs   = extract_intervals(gt_mask,   fs)
    pred_ivs = extract_intervals(pred_mask, fs)

    qrs_off = pred_ivs.get('QRS_Offset')
    t_on    = pred_ivs.get('T_Onset')
    t_off   = pred_ivs.get('T_Offset')
    has_st  = (qrs_off is not None and t_on is not None and t_on > qrs_off)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={'hspace': 0.35})
    fig.suptitle(f'Segmentazione ECG - {fname}   [derivazione: {lead}]',
                 fontsize=11, fontweight='bold', y=0.99)

    boundaries = ['P_Onset','P_Offset','QRS_Onset','QRS_Offset','T_Onset','T_Offset']

    ax0 = axes[0]
    ax0.plot(t_ms, sig, color='#212121', linewidth=0.85, zorder=5)
    shade(ax0, gt_ivs.get('P_Onset'),   gt_ivs.get('P_Offset'),  C_P,   0.35, 'Onda P')
    shade(ax0, gt_ivs.get('QRS_Onset'), gt_ivs.get('QRS_Offset'),C_QRS, 0.35, 'QRS')
    shade(ax0, gt_ivs.get('T_Onset'),   gt_ivs.get('T_Offset'),  C_TGT, 0.35, 'Onda T (incl. ST)')
    for k in boundaries: vline(ax0, gt_ivs.get(k))
    ax0.set_title('->  Ground Truth GE   |   Nota: T ingloba ST (T_Onset = QRS_Offset)',
                  fontsize=8.5, loc='left', pad=3, color='#555')
    ax0.set_ylabel('Ampiezza (mV)', fontsize=9)
    ax0.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=3)
    ax0.grid(True, alpha=0.22, linestyle='--')

    ax1 = axes[1]
    ax1.plot(t_ms, sig, color='#212121', linewidth=0.85, zorder=5)
    shade(ax1, pred_ivs.get('P_Onset'),   pred_ivs.get('P_Offset'),  C_P,   0.35, 'Onda P')
    shade(ax1, pred_ivs.get('QRS_Onset'), pred_ivs.get('QRS_Offset'),C_QRS, 0.35, 'QRS')
    if has_st:
        shade(ax1, qrs_off, t_on, C_ST, 0.55, f'ST  ({t_on - qrs_off:.0f} ms)')
    shade(ax1, pred_ivs.get('T_Onset'), pred_ivs.get('T_Offset'), C_T, 0.40, 'Onda T (pura)')

    if has_st and t_off is not None:
        ax1.axvspan(qrs_off, t_off, alpha=0.07, color=C_TGT, zorder=1)
        for xv in (qrs_off, t_off): ax1.axvline(xv, color=C_TGT, linewidth=1.6, linestyle='--', zorder=7, alpha=0.85)
        tgt_patch = mpatches.Patch(facecolor=C_TGT, alpha=0.4, label='T estesa (ST+T) ≈ GT T')

    for k in boundaries: vline(ax1, pred_ivs.get(k))
    if has_st:
        for xv in (qrs_off, t_on): ax1.axvline(xv, color='#F9A825', linewidth=1.0, linestyle='--', zorder=6, alpha=0.9)

    ax1.set_title('->  Predizione Ensemble (Fase2 V3 + Fase6)', fontsize=8.5, loc='left', pad=3, color='#555')
    ax1.set_ylabel('Ampiezza (mV)', fontsize=9)
    ax1.set_xlabel('Tempo (ms)', fontsize=9)
    ax1.grid(True, alpha=0.22, linestyle='--')

    handles, labels = ax1.get_legend_handles_labels()
    if has_st and t_off is not None: handles.append(tgt_patch)
    ax1.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9, ncol=4)

    fig.subplots_adjust(top=0.93, hspace=0.35)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_12_leads_comparison(signals, gt_mask, pred_mask, fname, output_path, fs=TARGET_FS):
    t_ms     = np.arange(MEDIAN_LEN) * 1000 / fs
    gt_ivs   = extract_intervals(gt_mask,   fs)
    pred_ivs = extract_intervals(pred_mask, fs)

    qrs_off = pred_ivs.get('QRS_Offset')
    t_on    = pred_ivs.get('T_Onset')
    has_st  = (qrs_off is not None and t_on is not None and t_on > qrs_off)

    fig, axes = plt.subplots(12, 2, figsize=(18, 38), sharex=True)
    fig.suptitle(f'CONFRONTO 12 DERIVAZIONI - {fname}', fontsize=16, fontweight='bold', y=0.995)

    for i, lead in enumerate(LEADS_ORDER):
        sig = signals.get(lead, np.zeros(MEDIAN_LEN))
        
        # Col 0: GT
        ax0 = axes[i, 0]
        ax0.plot(t_ms, sig, color='#212121', lw=0.8)
        shade(ax0, gt_ivs.get('P_Onset'),   gt_ivs.get('P_Offset'),  C_P,   0.25)
        shade(ax0, gt_ivs.get('QRS_Onset'), gt_ivs.get('QRS_Offset'),C_QRS, 0.25)
        shade(ax0, gt_ivs.get('T_Onset'),   gt_ivs.get('T_Offset'),  C_TGT, 0.25)
        ax0.set_ylabel(f'{lead}', fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        ax0.grid(True, alpha=0.15)
        if i == 0: ax0.set_title('GE Ground Truth', fontsize=14, pad=10)

        # Col 1: Pred
        ax1 = axes[i, 1]
        ax1.plot(t_ms, sig, color='#212121', lw=0.8)
        shade(ax1, pred_ivs.get('P_Onset'),   pred_ivs.get('P_Offset'),  C_P,   0.25)
        shade(ax1, pred_ivs.get('QRS_Onset'), pred_ivs.get('QRS_Offset'),C_QRS, 0.25)
        if has_st: shade(ax1, qrs_off, t_on, C_ST, 0.45)
        shade(ax1, pred_ivs.get('T_Onset'), pred_ivs.get('T_Offset'), C_T, 0.35)
        ax1.grid(True, alpha=0.15)
        if i == 0: ax1.set_title('Predizione Ensemble', fontsize=14, pad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    plt.savefig(output_path, dpi=120)
    plt.close()


def postprocess_mask(pred_mask, fs=TARGET_FS):
    cleaned = pred_mask.copy()
    
    # Sotto-funzione per pulire una singola classe
    def clean_class(m, cid, max_gap_samples, min_len_samples):
        is_c = (m == cid).astype(int)
        
        # 1. Unisci i gap minori di max_gap_samples
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        
        if len(on) > 1:
            for i in range(len(off) - 1):
                gap = on[i+1] - off[i]
                if gap <= max_gap_samples:
                    m[off[i]:on[i+1]] = cid
                    
        # Ricalcola on e off dopo l'unione
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        
        # 2. Tieni solo la componente più grande e scarta quelle troppo corte
        if len(on) > 0:
            lengths = off - on + 1
            max_idx = np.argmax(lengths)
            for i in range(len(on)):
                if i != max_idx or lengths[i] < min_len_samples:
                    m[on[i]:off[i]+1] = 0
        return m

    # Parametri (a 500Hz: 20 campioni = 40ms)
    # QRS: unisce buchi fino a 30ms, lunghezza min 40ms (20 campioni)
    cleaned = clean_class(cleaned, 2, max_gap_samples=15, min_len_samples=20)
    # P: unisce buchi fino a 50ms, lunghezza min 10ms (5 campioni)
    cleaned = clean_class(cleaned, 1, max_gap_samples=25, min_len_samples=5)
    # T: unisce buchi fino a 80ms, rimuove limite di lunghezza minima
    cleaned = clean_class(cleaned, 3, max_gap_samples=40, min_len_samples=0)

    # (Nessuna regola basata su GT o Diagnosi testuali per mantenere il modello 100% autonomo end-to-end)

    # 4. Regole fisiologiche (P prima di QRS, T dopo QRS)
    idx = np.arange(len(cleaned))
    ivs = extract_intervals(cleaned, fs)
    qrs_on, qrs_off = ivs.get("QRS_Onset"), ivs.get("QRS_Offset")
    
    # Regola T: la T non può iniziare molto prima della fine del QRS.
    # Usiamo una tolleranza di 50ms per non eliminare T nei battiti GE/HYPER
    # dove il QRS è largo e la T inizia subito dopo il picco S.
    if qrs_off is not None:
        tolerance_ms = 50  # ms di tolleranza
        tolerance_s = int(tolerance_ms * fs / 1000)
        qrs_off_s = int(qrs_off * fs / 1000)
        # Elimina solo la T che inizia PIÙ di 50ms prima della fine del QRS
        cleaned[(cleaned == 3) & (idx < max(0, qrs_off_s - tolerance_s))] = 0
    # Regola P: la P non può estendersi oltre l'inizio del QRS
    if qrs_on is not None:
        qrs_on_s = int(qrs_on * fs / 1000)
        cleaned[(cleaned == 1) & (idx >= qrs_on_s)] = 0
        
    return cleaned

def run_analysis(xml_files, model_v3_path, model_fase6_path, output_csv="test_results.csv", dataset_pt_path=None, img_dir_name='risultati_test_xml_post'):
    """
    Esegue l'analisi su una lista di percorsi XML.
    dataset_pt_path: percorso al file .pt per estrarre lo split (Train/Val/Test).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  ECG SEGMENTATION TEST — Ensemble Fase2 V3 + Fase6")
    print(f"{'='*70}")
    print(f"  Device:       {device}")
    print(f"  Model Fase2 V3: {model_v3_path}\n  Model F6:       {model_fase6_path}")
    print(f"  File XML:     {len(xml_files)}")
    
    # Mappa dello split se fornita
    split_mapping = {}
    if dataset_pt_path and os.path.exists(dataset_pt_path):
        print(f"  Dataset .pt:  {dataset_pt_path}")
        try:
            data = torch.load(dataset_pt_path, map_location='cpu', weights_only=False)
            record_metadata = data.get('record_metadata', {})
            # Costruiamo il set di RID per ogni split
            train_recs = set([data['record_ids'][i] for i in data['train_indices']])
            val_recs   = set([data['record_ids'][i] for i in data['val_indices']])
            test_recs  = set([data['record_ids'][i] for i in data['test_indices']])
            
            for rid, meta in record_metadata.items():
                fname = meta['xml_file']
                if rid in train_recs: split = 'Train'
                elif rid in val_recs: split = 'Val'
                elif rid in test_recs: split = 'Test'
                else: split = 'Unknown'
                split_mapping[fname] = split
            print(f"  -> Split caricato: {len(split_mapping)} record mappati.")
        except Exception as e:
            print(f"  [!] Errore nel caricamento del dataset .pt: {e}")
    
    print(f"  Output CSV:   {output_csv}")
    
    output_dir = os.path.join(os.path.dirname(output_csv), img_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output Immagini: {output_dir}")


    # Carico modello Fase2 V3 (P e QRS)
    model_v3 = EnhancedUNetV3(in_channels=1, num_classes=4, f=12).to(device)
    model_v3.load_state_dict(torch.load(model_v3_path, map_location=device, weights_only=True))
    model_v3.eval()
    
    # Carico modello Fase6 (T e ST)
    model_f6 = ECGUNetFase6(in_channels=1, num_classes=4).to(device)
    ckpt_f6 = torch.load(model_fase6_path, map_location=device, weights_only=False)
    if isinstance(ckpt_f6, dict) and 'model_state_dict' in ckpt_f6:
        ckpt_f6 = ckpt_f6['model_state_dict']
    # Rimuovi eventuale 'backbone.' prefix
    cleaned_f6 = {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k: v for k, v in ckpt_f6.items()}
    model_f6.load_state_dict(cleaned_f6, strict=False)
    model_f6.eval()
    print("  Modelli Ensemble caricati con successo.")



    results = []
    results_per_lead = []
    processed, skipped = 0, 0

    for xml_path in xml_files:
        fname = os.path.basename(xml_path)
        parsed = parse_xml(xml_path)

        if parsed is None:
            print(f"  [SKIP] {fname} — formato non riconosciuto o nessun median beat")
            skipped += 1
            continue

        signals, ann, statements, per_lead_ann = parsed
        macro_class, has_artifact, is_path_strict, is_path_lax = get_macro_class(statements)
        file_split = split_mapping.get(fname, 'Unknown')

        if 'Q_Onset' not in ann:
            print(f"  [SKIP] {fname} — annotazioni incomplete (manca Q_Onset)")
            skipped += 1
            continue

        gt_mask = build_mask(ann)

        # Inferenza: ogni derivazione separatamente + media finale
        lead_scores_v3 = []
        lead_scores_f6 = []
        with torch.no_grad():
            for lead in LEADS_ORDER:
                raw_sig = signals[lead] if lead in signals else np.zeros(MEDIAN_LEN, dtype=np.float32)
                filt_sig = apply_ecg_filters(raw_sig, TARGET_FS)
                sig_norm = normalize(filt_sig)
                x = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                scores_v3 = torch.softmax(model_v3(x), dim=1).squeeze(0)
                scores_f6 = torch.softmax(model_f6(x), dim=1).squeeze(0)
                
                lead_scores_v3.append((lead, scores_v3))
                lead_scores_f6.append((lead, scores_f6))

        for (lead, scores_v3), (_, scores_f6) in zip(lead_scores_v3, lead_scores_f6):
            # Per l'estrazione delle ampiezze usiamo il segnale FILTRATO (clinicamente più corretto)
            raw_sig = signals[lead] if lead in signals else np.zeros(MEDIAN_LEN, dtype=np.float32)
            sig_for_amp = apply_ecg_filters(raw_sig, TARGET_FS)

            # ENSEMBLE LOGIC PER SINGOLA DERIVAZIONE
            pred_v3 = scores_v3.argmax(0).cpu().numpy()
            pred_f6 = scores_f6.argmax(0).cpu().numpy()
            
            pred_lead = np.zeros(MEDIAN_LEN, dtype=np.int64)
            pred_lead[pred_f6 == 3] = 3  # T da Fase6
            pred_lead[pred_v3 == 1] = 1  # P da V3
            pred_lead[pred_v3 == 2] = 2  # QRS da V3
            
            pred_lead = postprocess_mask(pred_lead, fs=TARGET_FS)

            f1_lead   = compute_sample_f1(pred_lead, gt_mask)
            te_lead   = compute_timing_errors(pred_lead, gt_mask)
            pi_lead   = extract_intervals(pred_lead)
            
            # Calcolo picchi IA e picchi GE (GT)
            amp_pred  = extract_amplitudes(sig_for_amp, pred_lead)
            amp_gt_vals = extract_amplitudes(sig_for_amp, gt_mask)
            
            # Estrazione ampiezze GT ufficiali dall'XML (per confronto y)
            gt_amp_xml = per_lead_ann.get(lead, {})
            gt_amps_mv = {
                'gt_P_Amp': gt_amp_xml.get('P_Amp') / 1000.0 if gt_amp_xml.get('P_Amp') is not None else None,
                'gt_Q_Amp': gt_amp_xml.get('Q_Amp') / 1000.0 if gt_amp_xml.get('Q_Amp') is not None else None,
                'gt_R_Amp': gt_amp_xml.get('R_Amp') / 1000.0 if gt_amp_xml.get('R_Amp') is not None else None,
                'gt_S_Amp': gt_amp_xml.get('S_Amp') / 1000.0 if gt_amp_xml.get('S_Amp') is not None else None,
                'gt_J_Amp': gt_amp_xml.get('STJ_Amp') / 1000.0 if gt_amp_xml.get('STJ_Amp') is not None else None,
                'gt_T_Amp': gt_amp_xml.get('T_Amp') / 1000.0 if gt_amp_xml.get('T_Amp') is not None else None,
            }
            
            # Aggiungiamo le X GT (posizioni temporali calcolate nel recinto GE)
            gt_peaks_x = {f"gt_{k.replace('pred_', '')}": v for k, v in amp_gt_vals.items() if '_Peak_ms' in k}

            results_per_lead.append({
                'file': fname, 'lead': lead, 'split': file_split,
                'macro_class': macro_class, 'has_artifact': has_artifact,
                'is_path_strict': is_path_strict, 'is_path_lax': is_path_lax,
                **f1_lead, **te_lead,
                **{f'pred_{k}': v for k, v in pi_lead.items()},
                **{f'pred_{k}': v for k, v in amp_pred.items()},
                **gt_amps_mv,
                **gt_peaks_x
            })

        # Predizione media sulle 12 derivazioni
        avg_v3 = torch.stack([s for _, s in lead_scores_v3]).mean(0)
        avg_f6 = torch.stack([s for _, s in lead_scores_f6]).mean(0)
        
        pred_v3_glob = avg_v3.argmax(0).cpu().numpy()
        pred_f6_glob = avg_f6.argmax(0).cpu().numpy()
        
        pred_mask = np.zeros(MEDIAN_LEN, dtype=np.int64)
        pred_mask[pred_f6_glob == 3] = 3
        pred_mask[pred_v3_glob == 1] = 1
        pred_mask[pred_v3_glob == 2] = 2
        
        pred_mask = postprocess_mask(pred_mask, fs=TARGET_FS)
        
        f1_stats    = compute_sample_f1(pred_mask, gt_mask)
        timing_errs = compute_timing_errors(pred_mask, gt_mask)
        pred_intervals = extract_intervals(pred_mask)
        
        # Per la riga globale prendiamo le ampiezze basate sul segnale DII FILTRATO
        ref_raw = signals['II'] if 'II' in signals else np.zeros(MEDIAN_LEN, dtype=np.float32)
        ref_filt = apply_ecg_filters(ref_raw, TARGET_FS)
        global_amps = extract_amplitudes(ref_filt, pred_mask)

        results.append({
            'file': fname, 'split': file_split,
            'macro_class': macro_class, 'has_artifact': has_artifact,
            'is_path_strict': is_path_strict, 'is_path_lax': is_path_lax,
            **f1_stats, **timing_errs,
            **{f'pred_{k}': v for k, v in pred_intervals.items()},
            **{f'ref_II_{k}': v for k, v in global_amps.items()}
        })
        processed += 1

        print(f"  [{processed:3d}] {fname}")
        print(f"        F1 Macro={f1_stats['f1_macro']:.1f}%  "
              f"(P:{f1_stats['f1_c1']:.1f}  QRS:{f1_stats['f1_c2']:.1f}  T:{f1_stats['f1_c3']:.1f})")
        for key in ['P_Onset','P_Offset','QRS_Onset','QRS_Offset','T_Onset','T_Offset']:
            err = timing_errs.get(f'err_{key}_ms')
            if err is not None:
                judgment = get_clinical_judgment(err)
                msg = f"        {key:>12}: err = {err:+6.1f} ms  [{judgment}]"
                if key == 'T_Onset' and judgment == "Fuori Tolleranza":
                    msg += "  (Atteso/Normale: la rete isola correttamente l'ST, il GT no)"
                print(msg)
                
        # Salva SOLO immagine plot (tutte le 12 derivazioni)
        img_12_name = os.path.splitext(fname)[0] + '_12leads.png'
        img_12_path = os.path.join(output_dir, img_12_name)
        plot_12_leads_comparison(signals, gt_mask, pred_mask, fname, img_12_path)
        
        print(f"        -> Immagine salvata: {img_12_name}")

    #riepilogo
    if not results:
        print("\n  [!] Nessun file processato con successo.")
        return None

    df = pd.DataFrame(results)
    df_per_lead = pd.DataFrame(results_per_lead)

    print(f"\n{'='*70}")
    print(f"  RIEPILOGO GLOBALE — {processed} file analizzati ({skipped} saltati)")
    print(f"{'='*70}")
    print(f"  F1 Macro medio:     {df['f1_macro'].mean():.2f}%")
    print(f"  F1 Onda P medio:    {df['f1_c1'].mean():.2f}%")
    print(f"  F1 QRS medio:       {df['f1_c2'].mean():.2f}%")
    print(f"  F1 Onda T medio:    {df['f1_c3'].mean():.2f}%  (Atteso: la rete separa ST, il GT lo include in T)")
    print(f"  Accuracy medio:     {df['accuracy'].mean():.2f}%")

    print(f"\n  {'Punto':<15} {'Errore Medio':>14} {'Std':>10} {'Giudizio':<20}")
    print(f"  {'-'*60}")
    for key in ['P_Onset','P_Offset','QRS_Onset','QRS_Offset','T_Onset','T_Offset']:
        col = f'err_{key}_ms'
        vals = df[col].dropna()
        if len(vals) > 0:
            m, s = vals.mean(), vals.std()
            judgment = get_clinical_judgment(m)
            msg = f"  {key:<15} {m:>+10.1f} ms  {s:>8.1f} ms  [{judgment}]"
            if key == 'T_Onset' and judgment == "Fuori Tolleranza":
                msg += "  <-- NORMALE: La rete riconosce l'ST isolato, il GT lo ingloba."
            print(msg)

    print(f"\n  {'Intervallo':<15} {'Errore Medio':>14} {'Std':>10}")
    print(f"  {'-'*45}")
    for key in ['PR_Interval','QT_Interval','P_Duration','QRS_Duration']:
        col = f'err_{key}_ms'
        vals = df[col].dropna()
        if len(vals) > 0:
            m, s = vals.mean(), vals.std()
            print(f"  {key:<15} {m:>+10.1f} ms  {s:>8.1f} ms")

    print(f"\n  {'Parametro':<15} {'IA (x)':>10} {'GE (x)':>10} {'IA (y)':>10} {'GE (y)':>10} {'Err Abs (y)':>12}")
    print(f"  {'-'*80}")
    for key in ['P_Amp', 'Q_Amp', 'R_Amp', 'S_Amp', 'J_Amp', 'T_Amp']:
        pred_col = f'pred_{key}'
        gt_col = f'gt_{key}'
        pred_x_col = f"pred_{key.replace('_Amp', '_Peak_ms')}"
        gt_x_col   = f"gt_{key.replace('_Amp', '_Peak_ms')}"
        
        if pred_col in df_per_lead.columns and gt_col in df_per_lead.columns:
            valid_idx = df_per_lead[pred_col].notna() & df_per_lead[gt_col].notna()
            if valid_idx.sum() > 0:
                mean_pred_y = df_per_lead.loc[valid_idx, pred_col].mean()
                mean_gt_y   = df_per_lead.loc[valid_idx, gt_col].mean()
                mean_pred_x = df_per_lead.loc[valid_idx, pred_x_col].mean() if pred_x_col in df_per_lead.columns else 0
                mean_gt_x   = df_per_lead.loc[valid_idx, gt_x_col].mean() if gt_x_col in df_per_lead.columns else 0
                mae_y = (df_per_lead.loc[valid_idx, pred_col] - df_per_lead.loc[valid_idx, gt_col]).abs().mean()
                
                name = key.replace('_Amp', '')
                print(f"  {name:<15} {mean_pred_x:>7.1f} ms {mean_gt_x:>7.1f} ms {mean_pred_y:>7.3f} mV {mean_gt_y:>7.3f} mV {mae_y:>10.3f} mV")

    #save
    df.to_csv(output_csv, index=False)
    # CSV per derivazione: stesso nome base con suffisso _per_lead
    per_lead_csv = output_csv.replace('.csv', '_per_lead.csv')
    if not per_lead_csv.endswith('_per_lead.csv'):
        per_lead_csv = output_csv + '_per_lead.csv'
    df_per_lead.to_csv(per_lead_csv, index=False)
    print(f"  CSV per derivazione: {os.path.abspath(per_lead_csv)}")
    print(f"\n  [SAVED] {os.path.abspath(output_csv)}")
    print(f"{'='*70}\n")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Testa un modello Enhanced U-Net su file XML GE"
    )
    parser.add_argument("--xml-dir", required=True,
                        help="Cartella contenente i file XML")
    parser.add_argument("--model-v3", 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_fase2_v3.pt"),
                        help="Percorso rete Fase2 V3 (P e QRS)")
    parser.add_argument("--model-fase6", 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_fase6.pt"),
                        help="Percorso rete Fase6 (T e ST)")
    parser.add_argument("--output-csv", default="test_results.csv",
                        help="Nome del file CSV di output")
    parser.add_argument("--dataset-pt", 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "12 dev", "Total", "timbro_12dev_dataset.pt"),
                        help="Percorso del file .pt del dataset per mappare Train/Val/Test")
    parser.add_argument("--img-dir", default="risultati_xml_derivazioni",
                        help="Nome della cartella dove salvare le immagini")
    args = parser.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "**", "*.xml"), recursive=True))
    if not xml_files:
        print(f"  [ERRORE] Nessun file XML trovato in: {args.xml_dir}")
        return

    run_analysis(xml_files, args.model_v3, args.model_fase6, args.output_csv, args.dataset_pt, args.img_dir)


if __name__ == "__main__":
    main()