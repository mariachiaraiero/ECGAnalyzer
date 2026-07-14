import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from test_ensemble_fasi_post import EnhancedUNetV3, ECGUNetFase6
def postprocess_mask_multibeat(mask, fs=500):
    """Post-processing adattato per file multi-battito (LUDB).
    Non cancella le componenti multiple, ma unisce solo i buchi 
    e rimuove i frammenti troppo piccoli."""
    cleaned = np.copy(mask)
    
    def clean_class_multibeat(m, cid, max_gap_samples, min_len_samples):
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
                    
        # 2. Ricalcola on e off e scarta frammenti corti (NESSUN MAX_IDX)
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        
        if len(on) > 0:
            lengths = off - on + 1
            for i in range(len(on)):
                if lengths[i] < min_len_samples:
                    m[on[i]:off[i]+1] = 0
        return m

    # Parametri: QRS (30ms gap, soglia 10ms), P (50ms gap, 10ms), T (80ms gap, 0ms)
    cleaned = clean_class_multibeat(cleaned, 2, max_gap_samples=15, min_len_samples=5)
    cleaned = clean_class_multibeat(cleaned, 1, max_gap_samples=10, min_len_samples=5)
    cleaned = clean_class_multibeat(cleaned, 3, max_gap_samples=40, min_len_samples=0)

    # --- SEZIONI A/B/C/D RIPRISTINATE ---
    def get_intervals(m, cid):
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        return on, off

    qrs_on, qrs_off = get_intervals(cleaned, 2)

    # A. Periodo refrattario QRS (max 250 ms)
    min_qrs_dist = int(250 * fs / 1000)
    if len(qrs_on) > 1:
        keep_qrs = [0]
        for i in range(1, len(qrs_on)):
            if qrs_on[i] - qrs_on[keep_qrs[-1]] >= min_qrs_dist:
                keep_qrs.append(i)
            else:
                len_prev = qrs_off[keep_qrs[-1]] - qrs_on[keep_qrs[-1]]
                len_curr = qrs_off[i] - qrs_on[i]
                if len_curr > len_prev:
                    cleaned[qrs_on[keep_qrs[-1]]:qrs_off[keep_qrs[-1]]+1] = 0
                    keep_qrs[-1] = i
                else:
                    cleaned[qrs_on[i]:qrs_off[i]+1] = 0
    qrs_on, qrs_off = get_intervals(cleaned, 2)

    # B. Periodo refrattario P (max 200 ms)
    p_on, p_off = get_intervals(cleaned, 1)
    min_p_dist = int(200 * fs / 1000)
    if len(p_on) > 1:
        keep_p = [0]
        for i in range(1, len(p_on)):
            if p_on[i] - p_on[keep_p[-1]] >= min_p_dist:
                keep_p.append(i)
            else:
                len_prev = p_off[keep_p[-1]] - p_on[keep_p[-1]]
                len_curr = p_off[i] - p_on[i]
                if len_curr > len_prev:
                    cleaned[p_on[keep_p[-1]]:p_off[keep_p[-1]]+1] = 0
                    keep_p[-1] = i
                else:
                    cleaned[p_on[i]:p_off[i]+1] = 0

    # C. Onda T orfana: non può esserci una T se non c'è un QRS che termini
    # nei precedenti 1200 ms. Usiamo q_off (fine QRS) per gestire QRS larghi (GE/HYPER).
    t_on, t_off = get_intervals(cleaned, 3)
    max_t_delay = int(1200 * fs / 1000)
    for i in range(len(t_on)):
        valid_qrs = [q_off for q_on, q_off in zip(qrs_on, qrs_off)
                     if q_off <= t_on[i] + 50 and (t_on[i] - q_off) <= max_t_delay]
        if not valid_qrs:
            cleaned[t_on[i]:t_off[i]+1] = 0

    # D. Onda P duplicata (tieni solo ultima P prima di ogni QRS)
    p_on, p_off = get_intervals(cleaned, 1)
    if len(p_on) > 0 and len(qrs_on) > 0:
        valid_p_indices = set()
        for i, q_on in enumerate(qrs_on):
            prev_q_off = 0 if i == 0 else qrs_off[i-1]
            p_in_interval = [j for j in range(len(p_on)) if p_on[j] >= prev_q_off and p_off[j] <= q_on]
            if p_in_interval:
                valid_p_indices.add(p_in_interval[-1])
        for j in range(len(p_on)):
            if j not in valid_p_indices:
                cleaned[p_on[j]:p_off[j]+1] = 0

    return cleaned

