"""
trova_picchi_patologie.py

Applica il modello Ensemble2+6 (Fase2 V3 per P/QRS + Fase6 per T/ST) ai file XML
GE MAC2000, poi cerca i picchi P, Q, R, S, T, J dentro gli intervalli predetti
usando algoritmi di peak detection specifici per macro-classe patologica.

Struttura del file:
  1. Import e costanti globali
  2. Architetture dei modelli (EnhancedUNetV3, ECGUNetFase6)
  3. Mappe cliniche (classi patologiche, polarita' attese per derivazione)
  4. Parsing XML e preprocessing del segnale
  5. Segmentazione: maschere, intervalli, postprocessing
  6. Helper condivisi per il peaka detection
  7. Detector QRS per macro-classe
  8. Detector P, T, J
  9. Dispatch dei detector (gestione multi-patologia)
  10. Plotting
  11. Pipeline principale (run)
"""

# =====================================================================
# 1. IMPORT E COSTANTI GLOBALI
# =====================================================================

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, medfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_XML_ROOT_DIR = "/content/drive/MyDrive/TESI/XML patologie"
DEFAULT_MODEL_DIR = "/content/drive/MyDrive/TESI/ensemble finale fasi 2+6"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/TESI/ensemble finale fasi 2+6/plot_picchi_patologie"

# Ordine standard delle 12 derivazioni
LEADS_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Il battito mediano GE MAC2000 e' sempre lungo 600 campioni a 500 Hz
MEDIAN_LEN = 600
TARGET_FS = 500

# Colori per il plotting: P, QRS, ST, T (predizione), T (ground truth)
C_P, C_QRS, C_ST, C_T, C_TGT = '#2196F3', '#E53935', '#FFC107', '#43A047', '#FF7043'

# Derivazioni in cui ha senso misurare il punto J e la deviazione del tratto ST
J_POINT_LEADS = {"V1", "V2", "V3", "V4", "V5", "V6", "AVR"}


# =====================================================================
# 2. ARCHITETTURE DEI MODELLI
# =====================================================================

class SEBlock(nn.Module):
    # Squeeze-and-Excitation: ripesa i canali in base al loro contenuto globale
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(x.mean(-1)).view(b, c, 1)


class AttBlock(nn.Module):
    # Blocco convoluzionale residuo con attenzione sui canali (SEBlock)
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
    # Estrae feature a tre scale temporali diverse e le concatena
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
    # Modello Fase2: usato per la segmentazione di P e QRS
    def __init__(self, in_channels=1, num_classes=4, f=12, dropout=0.15):
        super().__init__()
        self.ms = MultiScaleInput(in_channels, f)
        self.enc1 = AttBlock(f, f, dropout)
        self.enc2 = AttBlock(f, f * 2, dropout)
        self.enc3 = AttBlock(f * 2, f * 4, dropout)
        self.enc4 = AttBlock(f * 4, f * 8, dropout)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = AttBlock(f * 8, f * 16, dropout)
        self.up = nn.ModuleList([nn.ConvTranspose1d(f * i, f * i, 8, 2, 3)
                                 for i in [16, 8, 4, 2]])
        self.dec = nn.ModuleList([AttBlock(f * 24, f * 8, dropout),
                                  AttBlock(f * 12, f * 4, dropout),
                                  AttBlock(f * 6, f * 2, dropout),
                                  AttBlock(f * 3, f, dropout)])
        self.final = nn.Conv1d(f, num_classes, 1)
        self.ds4 = nn.Conv1d(f * 8, num_classes, 1)
        self.ds3 = nn.Conv1d(f * 4, num_classes, 1)

    def forward(self, x):
        ms = self.ms(x)
        e1 = self.enc1(ms)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        def pm(xu, xs):
            return F.interpolate(xu, size=xs.size(2))

        d4 = self.dec[0](torch.cat([pm(self.up[0](b), e4), e4], 1))
        d3 = self.dec[1](torch.cat([pm(self.up[1](d4), e3), e3], 1))
        d2 = self.dec[2](torch.cat([pm(self.up[2](d3), e2), e2], 1))
        d1 = self.dec[3](torch.cat([pm(self.up[3](d2), e1), e1], 1))
        return self.final(d1)


class MultiScaleInputFase6(nn.Module):
    # Variante multi-scala della Fase6: interpolazione lineare invece che nearest
    def __init__(self, in_ch=1, out_ch=12):
        super().__init__()
        c = out_ch // 3
        self.s1 = nn.Sequential(nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(nn.AvgPool1d(kernel_size=2, stride=2), nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(nn.AvgPool1d(kernel_size=4, stride=4), nn.Conv1d(in_ch, c, kernel_size=5, padding=2), nn.BatchNorm1d(c), nn.ReLU(inplace=True))

    def forward(self, x):
        L = x.size(2)
        return torch.cat([self.s1(x),
                          F.interpolate(self.s2(x), size=L, mode='linear', align_corners=False),
                          F.interpolate(self.s3(x), size=L, mode='linear', align_corners=False)], dim=1)


class ECGUNetFase6(nn.Module):
    # Modello Fase6: knowledge distillation su tracce raw, usato per l'onda T.
    # Corregge la convenzione GE che impone T_Onset = QRS_Offset.
    def __init__(self, in_channels=1, num_classes=4, base_filters=12, dropout=0.15):
        super().__init__()
        f = base_filters
        self.ms = MultiScaleInputFase6(in_channels, f)
        self.enc1 = AttBlock(f, f, dropout=dropout)
        self.enc2 = AttBlock(f, f * 2, dropout=dropout)
        self.enc3 = AttBlock(f * 2, f * 4, dropout=dropout)
        self.enc4 = AttBlock(f * 4, f * 8, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = AttBlock(f * 8, f * 16, dropout=dropout)
        self.up = nn.ModuleList([
            nn.ConvTranspose1d(f * 16, f * 16, kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f * 8, f * 8, kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f * 4, f * 4, kernel_size=8, stride=2, padding=3),
            nn.ConvTranspose1d(f * 2, f * 2, kernel_size=8, stride=2, padding=3),
        ])
        self.dec = nn.ModuleList([
            AttBlock(f * 24, f * 8, dropout=dropout),
            AttBlock(f * 12, f * 4, dropout=dropout),
            AttBlock(f * 6, f * 2, dropout=dropout),
            AttBlock(f * 3, f, dropout=dropout),
        ])
        self.final = nn.Conv1d(f, num_classes, kernel_size=1)
        self.ds4 = nn.Conv1d(f * 8, num_classes, kernel_size=1)
        self.ds3 = nn.Conv1d(f * 4, num_classes, kernel_size=1)

    @staticmethod
    def _pad_to_match(x, n):
        # Allinea la lunghezza di x a n con padding simmetrico o troncamento
        d = n - x.size(2)
        if d > 0:
            return F.pad(x, (d // 2, d - d // 2))
        if d < 0:
            return x[:, :, :n]
        return x

    def forward(self, x):
        ms = self.ms(x)
        e1 = self.enc1(ms)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        skips = [e4, e3, e2, e1]
        d = b
        for up, dec_block, skip in zip(self.up, self.dec, skips):
            d = up(d)
            d = self._pad_to_match(d, skip.size(2))
            d = torch.cat([d, skip], dim=1)
            d = dec_block(d)
        return self.final(d)


# =====================================================================
# 3. MAPPE CLINICHE
# =====================================================================

# Ogni macro-classe e' definita dalle diciture diagnostiche GE che la attivano.
# Le stringhe sono confrontate in lowercase con il testo degli statement XML.
CLASS_MAPPING = {
    'PACING': ['pacing atriale', 'pacing ventricolare', 'stimolazione', 'pacemaker'],
    'NO_P': ['fibrillazione atriale', 'flutter atriale', 'ritmo giunzionale',
             'ritmo idioventricolare', 'centro giunzionale'],
    'WIDE_QRS': ['blocco di branca', 'emiblocco', 'allargamento del qrs',
                 'allargamento qrs', 'conduzione intraventricolare',
                 'blocco intraventricolare', 'ritardo della conduzione'],
    'OLD_MI': ['infarto', 'onda q patolog'],
    'AV_BLOCK': ['blocco a-v', 'bav', 'atrio-ventricolare', 'atrioventricolare',
                 'p-r corto', 'pr corto'],
    'ST_T': ['st aspecifiche', 'anormalit', 'ischemia', 'depressione st',
             'sottoslivellamento', 'sopraslivellamento', 'ripolarizzazione',
             'sovraccarico', 'pericardite', 'miocardite', 'qt lungo',
             'allungamento del qt'],
    'HYPERTROPHY': ['ipertrofia', 'ingrandimento atriale', 'ivs',
                    'deviazione assiale sinistra', 'deviazione assiale destra',
                    'basso voltaggio'],
    'HEALTHY': ['ecg normale', 'ritmo sinusale normale', 'bradicardia sinusale',
                'tachicardia sinusale', 'aritmia sinusale', 'peraltro ecg normale'],
}

# Diciture che segnalano solo rumore o artefatti: non definiscono una patologia
ARTIFACT_KEYWORDS = [
    'tremore muscolare', 'interferenza linea', 'interferenza della linea',
    'rumore elettrodi', 'qualita dati scadente', 'baseline wander', 'disturbo linea base',
]

# Gerarchia usata SOLO per assegnare l'etichetta principale del file (nome cartella,
# titolo del plot, stratificazione dei risultati). NON governa piu' la scelta dei
# detector: quella e' fatta da select_detectors(), che assegna il detector di ogni
# onda in modo indipendente.
PRIORITY = ['PACING', 'NO_P', 'WIDE_QRS', 'OLD_MI', 'AV_BLOCK', 'ST_T', 'HYPERTROPHY', 'HEALTHY']

# Polarita' attesa dell'onda P per derivazione (ritmo sinusale normale).
# E' una preferenza superabile, non una regola rigida: vedi _extremum_polarity.
P_POLARITY_PER_LEAD = {
    'I': 'pos', 'II': 'pos', 'III': 'pos',
    'AVF': 'pos',
    'V2': 'pos', 'V3': 'pos', 'V4': 'pos', 'V5': 'pos', 'V6': 'pos',
    'AVR': 'neg',
}

# Nel blocco AV e' alterata solo la conduzione, non l'origine dell'impulso, che resta
# il nodo del seno: la polarita' della P resta quindi quella fisiologica.
P_POLARITY_AVBLOCK = dict(P_POLARITY_PER_LEAD)
P_POLARITY_AVBLOCK.update({'II': 'pos', 'III': 'pos', 'AVF': 'pos'})

# Polarita' attesa dell'onda T per derivazione
T_POLARITY_PER_LEAD = {
    'I': 'pos', 'II': 'pos',
    'V3': 'pos', 'V4': 'pos', 'V5': 'pos', 'V6': 'pos',
    'AVR': 'neg',
}

# Derivazioni in cui la P e' spesso bifasica per anatomia (atrio destro e sinistro)
P_BIFASICA_LEADS = {"II", "V1"}

# Derivazioni in cui la P bifida (dilatazione atriale sinistra) e' visibile
P_BIFIDA_LEADS = {"I", "AVL", "II"}

# Derivazioni con P pulmonale (dilatazione atriale destra): soglia di ampiezza piu' bassa
P_PULMONALE_LEADS = {"II", "III", "AVF"}

# Derivazioni precordiali
PRECORDIALI = {"V1", "V2", "V3", "V4", "V5", "V6"}

# In queste derivazioni la P e' spesso piatta o atipica: soglia di prominenza piu' alta
P_FIRST_FALLBACK_LEADS = {"V1", "III", "AVL"}

# Derivazioni in cui una Q profonda e' patologica (necrosi settale)
Q_ANOMALA_LEADS = {"V1", "V2", "V3"}

# Distanza massima tra le due fasi di una P bifasica (circa 60 ms a 500 Hz)
P_BIPHASIC_GAP_MAX_SAMPLES = 30


def get_matched_classes(statements):
    """Ritorna l'insieme di TUTTE le macro-classi che fanno match sugli statement.

    Non applica nessuna gerarchia: un ECG con blocco di branca, ipertrofia e
    alterazioni ST restituisce {'WIDE_QRS', 'HYPERTROPHY', 'ST_T'}. La scelta di quale
    detector usare per ciascuna onda e' delegata a select_detectors().
    """
    txt = ' '.join(statements).lower() if statements else ''
    matched = set()
    for mc, kws in CLASS_MAPPING.items():
        if any(kw in txt for kw in kws):
            matched.add(mc)
    return matched


def get_macro_class(statements):
    """Etichetta principale del file, scelta secondo PRIORITY.

    Serve solo a fini organizzativi: nome della cartella di output, titolo del plot,
    stratificazione dei risultati. I detector veri e propri sono scelti da
    select_detectors() sulla base dell'insieme COMPLETO delle classi attive, non di
    questa singola etichetta.
    """
    matched = get_matched_classes(statements)
    for p in PRIORITY:
        if p in matched:
            return p
    return 'UNKNOWN'


# =====================================================================
# 4. PARSING XML E PREPROCESSING
# =====================================================================

def parse_xml(xml_path):
    """Legge un XML GE MAC2000 ed estrae segnali, annotazioni e diagnosi.

    Ritorna la tupla (signals, global_ann, statements), oppure None se il file non e'
    nel formato atteso.
      signals     dict lead -> array di 600 campioni in mV
      global_ann  dict con P_Onset, Q_Onset, Q_Offset, T_Offset ecc. in millisecondi
      statements  lista delle diciture diagnostiche testuali del referto GE
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    # Il namespace principale e' dcar_1; alcuni file usano sapphire_3
    ns = {"ns": "urn:ge:sapphire:dcar_1"}
    all_wf = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)
    if not all_wf:
        ns = {"ns": "urn:ge:sapphire:sapphire_3"}
        all_wf = root.findall(".//ns:ecgWaveformMXG/ns:ecgWaveform", namespaces=ns)
    if not all_wf:
        return None

    # Il battito mediano e' identificato dalla lunghezza di 600 campioni
    median_wf = [wf for wf in all_wf
                 if wf.get('asizeVT', '0') == str(MEDIAN_LEN)
                 or wf.get('asizeBT', '0') == str(MEDIAN_LEN)]
    if not median_wf:
        return None

    signals = {}
    for wf in median_wf:
        lead = wf.get('lead', '').upper()
        vt_data = wf.get('V', '')
        if not lead or not vt_data:
            continue
        try:
            raw_vals = [int(v) for v in vt_data.split()]
        except ValueError:
            continue

        # -32768 e' il marcatore di campione mancante: si ripete l'ultimo valido
        clean_vals, last_valid = [], 0
        for v in raw_vals:
            if v != -32768:
                clean_vals.append(v)
                last_valid = v
            else:
                clean_vals.append(last_valid)

        # Forza la lunghezza esatta a MEDIAN_LEN
        if len(clean_vals) > MEDIAN_LEN:
            clean_vals = clean_vals[:MEDIAN_LEN]
        elif len(clean_vals) < MEDIAN_LEN:
            clean_vals += [last_valid] * (MEDIAN_LEN - len(clean_vals))

        # Conversione in mV: il fattore 4.88 uV/LSB e' proprio del MAC2000
        segnale = np.array([4.88 * v / 1000.0 for v in clean_vals], dtype=np.float32)

        # Alcuni record GE contengono il battito mediano ma con tutti i campioni a zero:
        # l'apparecchio non e' riuscito a calcolare il template, di solito perche' la
        # registrazione era troppo breve o troppo disturbata. La derivazione e' piatta e
        # non contiene segnale, quindi va scartata: su un array costante argmax e argmin
        # ritornano entrambi 0 e tutti i picchi collasserebbero sui primi campioni.
        if float(np.std(segnale)) < 1e-6:
            continue

        signals[lead] = segnale

    # Se nessuna derivazione contiene segnale, il record e' inutilizzabile
    if not signals:
        return None

    # Annotazioni globali del battito mediano, espresse in millisecondi
    global_ann = {}
    g_node = root.find(".//ns:medianTemplate/ns:measurements/ns:global", namespaces=ns)
    if g_node is not None:
        for child in g_node:
            v = child.get('V')
            if v and v != '-32768':
                tag = child.tag.split('}')[-1]
                try:
                    global_ann[tag] = int(v)
                except ValueError:
                    pass

    # Diciture diagnostiche del referto automatico GE: il testo sta nell'attributo V.
    # Sono queste a determinare la classe patologica, non il nome della cartella.
    statements = []
    for st in root.findall(".//ns:interpretation//ns:statement", namespaces=ns):
        txt = st.get('V')
        if txt:
            statements.append(txt)

    return signals, global_ann, statements


def normalize(sig):
    # Normalizzazione z-score, richiesta in input dalla rete
    std = np.std(sig)
    if std < 1e-6:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


def apply_ecg_filters(signal, fs=500):
    """Filtro notch a 50 Hz (rete elettrica) e passa-alto a 0.5 Hz (deriva di base)."""
    if np.all(signal == 0):
        return signal
    try:
        Q = 30.0
        w0 = 50.0 / (fs / 2)
        b_notch, a_notch = iirnotch(w0, Q)
        sig_notch = filtfilt(b_notch, a_notch, signal)

        cutoff = 0.5 / (fs / 2)
        b_hp, a_hp = butter(4, cutoff, btype='high')
        return filtfilt(b_hp, a_hp, sig_notch)
    except Exception:
        return signal


# =====================================================================
# 5. SEGMENTAZIONE: MASCHERE, INTERVALLI, POSTPROCESSING
# =====================================================================

def build_gt_mask(ann, fs=TARGET_FS):
    """Costruisce la maschera ground truth dalle annotazioni GE.

    Codifica: 0 = nessuna onda, 1 = P, 2 = QRS, 3 = T.

    Nota: GE impone per convenzione T_Onset = QRS_Offset, quindi il tratto ST non e'
    isolabile dalla sola ground truth. E' esattamente il problema che il modello di
    Fase6 risolve, imparando la convenzione clinica corretta dal dataset LUDB.
    """
    mask = np.zeros(MEDIAN_LEN, dtype=np.int64)

    def ms2s(ms):
        return max(0, min(int(ms * fs / 1000), MEDIAN_LEN - 1))

    p_on, p_off = ann.get('P_Onset'), ann.get('P_Offset')
    q_on, q_off = ann.get('Q_Onset'), ann.get('Q_Offset')
    t_off = ann.get('T_Offset')
    t_on = ann.get('T_Onset', ann.get('Q_Offset'))

    if p_on is not None and p_off is not None:
        s, e = ms2s(p_on), ms2s(p_off)
        if s < e:
            mask[s:e + 1] = 1
    if q_on is not None and q_off is not None:
        s, e = ms2s(q_on), ms2s(q_off)
        if s < e:
            mask[s:e + 1] = 2
    if t_on is not None and t_off is not None and t_off > t_on:
        s, e = ms2s(t_on), ms2s(t_off)
        if s < e:
            mask[s:e + 1] = 3

    return mask


def extract_intervals(mask, fs=500):
    """Estrae onset e offset di P, QRS e T da una maschera.

    IMPORTANTE: ritorna indici in CAMPIONI, non in millisecondi. Tutta la pipeline dei
    picchi lavora in campioni; la conversione in ms avviene solo nel plotting.
    """
    def get_pts(m, cid):
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1
        return (on[0] if len(on) > 0 else None,
                off[0] if len(off) > 0 else None)

    p_on, p_off = get_pts(mask, 1)
    q_on, q_off = get_pts(mask, 2)
    t_on, t_off = get_pts(mask, 3)

    return {
        'P_Onset': p_on, 'P_Offset': p_off,
        'QRS_Onset': q_on, 'QRS_Offset': q_off,
        'T_Onset': t_on, 'T_Offset': t_off,
    }


def postprocess_mask(pred_mask, fs=TARGET_FS):
    """Pulisce la maschera predetta dal modello.

    Per ciascuna classe P, QRS e T:
      1. Unisce i buchi piccoli: frammenti della stessa onda separati da un gap breve
         vengono ricongiunti, perche' il modello a volte spezza un'onda continua.
      2. Tiene solo il frammento contiguo piu' lungo e scarta i frammenti spuri sotto
         la lunghezza minima fisiologica.
      3. Applica i vincoli fisiologici: la T non puo' iniziare prima della fine del
         QRS, la P non puo' estendersi oltre l'inizio del QRS.
    """
    cleaned = pred_mask.copy()

    def clean_class(m, cid, max_gap_samples, min_len_samples):
        is_c = (m == cid).astype(int)

        # Passo 1: unisce i gap piu' corti di max_gap_samples
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1

        if len(on) > 1:
            for i in range(len(off) - 1):
                gap = on[i + 1] - off[i]
                if gap <= max_gap_samples:
                    m[off[i]:on[i + 1]] = cid

        # Ricalcola i frammenti dopo l'unione
        is_c = (m == cid).astype(int)
        diff = np.diff(np.pad(is_c, (1, 1), constant_values=0))
        on = np.where(diff == 1)[0]
        off = np.where(diff == -1)[0] - 1

        # Passo 2: tiene solo il frammento piu' lungo, se supera la soglia minima
        if len(on) > 0:
            lengths = off - on + 1
            max_idx = np.argmax(lengths)
            for i in range(len(on)):
                if i != max_idx or lengths[i] < min_len_samples:
                    m[on[i]:off[i] + 1] = 0
        return m

    # Parametri a 500 Hz: 20 campioni corrispondono a 40 ms
    # QRS: gap max 30 ms, durata minima 40 ms
    cleaned = clean_class(cleaned, 2, max_gap_samples=15, min_len_samples=20)
    # P: gap max 50 ms, durata minima 40 ms
    cleaned = clean_class(cleaned, 1, max_gap_samples=25, min_len_samples=20)
    # T: gap max 80 ms, durata minima 80 ms, perche' e' l'onda piu' larga
    cleaned = clean_class(cleaned, 3, max_gap_samples=40, min_len_samples=40)

    # Passo 3: vincoli fisiologici. Gli indici sono in campioni, quindi il confronto
    # con idx e' diretto e non serve nessuna conversione da millisecondi.
    idx = np.arange(len(cleaned))
    ivs = extract_intervals(cleaned, fs)
    qrs_on, qrs_off = ivs.get("QRS_Onset"), ivs.get("QRS_Offset")

    if qrs_off is not None:
        cleaned[(cleaned == 3) & (idx < qrs_off)] = 0
    if qrs_on is not None:
        cleaned[(cleaned == 1) & (idx >= qrs_on)] = 0

    return cleaned


# =====================================================================
# 6. HELPER CONDIVISI PER IL PEAK DETECTION
# =====================================================================

def _clip(idx, sig_len):
    # Vincola un indice dentro i limiti del segnale
    return int(max(0, min(sig_len - 1, idx)))


def _baseline(sig, ref_idx, window=3):
    # Baseline locale: media dei campioni intorno a ref_idx (tipicamente qrs_on)
    lo = _clip(ref_idx - window, len(sig))
    hi = _clip(ref_idx + window, len(sig))
    return float(np.mean(sig[lo:hi + 1]))


def _extremum(sig, start, end, mode="abs", baseline=0.0):
    """Indice dell'estremo nel tratto [start, end].

    mode: "max" massimo, "min" minimo, "abs" massima deviazione in valore assoluto.
    Ritorna None se il tratto e' degenere.
    """
    start, end = int(start), int(end)
    if end <= start or end - start < 1:
        return None
    segment = sig[start:end + 1] - baseline
    if mode == "max":
        rel = int(np.argmax(segment))
    elif mode == "min":
        rel = int(np.argmin(segment))
    else:
        rel = int(np.argmax(np.abs(segment)))
    return start + rel


def _local_extrema(sig, start, end, prominence_frac=0.10, min_prom_abs=0.015):
    """Massimi e minimi locali nel tratto, filtrati per prominenza.

    La prominenza minima e' proporzionale all'escursione del tratto, cosi' la soglia
    si adatta da sola alle derivazioni ad alto e a basso voltaggio.
    """
    start, end = int(start), int(end)
    if end <= start:
        return [], []
    segment = sig[start:end + 1]
    escursione = float(np.ptp(segment)) if len(segment) > 1 else 0.0
    prom = max(escursione * prominence_frac, min_prom_abs)
    max_idx, _ = find_peaks(segment, prominence=prom)
    min_idx, _ = find_peaks(-segment, prominence=prom)
    return [start + i for i in max_idx], [start + i for i in min_idx]


def _detect_pacing_spike(sig, on, off, max_width_samples=2, slope_factor=8.0,
                         return_tol_frac=0.35, min_slope=0.15,
                         allow_biphasic=False, biphasic_max_width=5, biphasic_min_amp=1.0):
    """Individua lo spike di stimolazione del pacemaker nel tratto [on, off].

    Uno spike vero e' strettissimo, da 1 a 3 campioni, e a pendenza altissima. Si
    cercano i punti con derivata sopra soglia, si raggruppano quelli contigui, e ogni
    gruppo e' validato su due criteri: larghezza sotto max_width_samples, e ritorno del
    segnale vicino al valore pre-spike, perche' un transiente vero va e torna.

    Parametro allow_biphasic, attivo solo per PACING: alcuni spike sono BIFASICI, cioe'
    scendono e risalgono violentemente in pochi campioni, per esempio da -2.6 mV a
    +5.3 mV. Queste oscillazioni consecutive formano un gruppo apparentemente largo che
    verrebbe scartato dal criterio di larghezza, e di conseguenza il picco R finirebbe
    erroneamente sullo spike. Con allow_biphasic il gruppo e' accettato se resta
    comunque stretto, entro biphasic_max_width, e la sua ampiezza e' molto grande,
    almeno biphasic_min_amp.

    Ritorna la coppia (start, end) dello spike piu' ampio trovato, oppure None.
    """
    on, off = int(on), int(off)
    if off <= on:
        return None
    segment = sig[on:off + 1]
    if len(segment) < 4:
        return None

    # Soglia adattiva sulla derivata, molto sopra la pendenza tipica del tratto
    deriv = np.abs(np.diff(segment))
    soglia = max(slope_factor * (np.median(deriv) + 1e-6), min_slope)
    candidati = np.where(deriv > soglia)[0]
    if len(candidati) == 0:
        return None

    # Raggruppa i campioni contigui sopra soglia in spike candidati distinti
    groups = []
    current_group = [candidati[0]]
    for c in candidati[1:]:
        if c - current_group[-1] <= 1:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)

    valid_spikes = []
    for g in groups:
        start_rel = g[0]
        end_rel = g[-1]
        larghezza = end_rel - start_rel

        if larghezza >= max_width_samples:
            # Ramo bifasico: gruppo largo ma comunque strettissimo e di ampiezza enorme
            if allow_biphasic and larghezza <= biphasic_max_width:
                picco_val = segment[start_rel:end_rel + 2]
                pre_idx = max(0, start_rel - 1)
                amp_big = float(np.max(np.abs(picco_val - segment[pre_idx])))
                if amp_big >= biphasic_min_amp:
                    valid_spikes.append((start_rel, end_rel, amp_big))
            continue

        # Ramo standard: verifica che il segnale ritorni vicino al valore pre-spike
        pre_idx = max(0, start_rel - 1)
        post_idx = min(len(segment) - 1, end_rel + 2)
        pre_val = segment[pre_idx]
        post_val = segment[post_idx]
        picco_val = segment[start_rel:end_rel + 2]
        ampiezza_spike = np.max(np.abs(picco_val - pre_val))
        ritorno = abs(post_val - pre_val)

        if ampiezza_spike >= 1e-9 and ritorno <= return_tol_frac * ampiezza_spike:
            valid_spikes.append((start_rel, end_rel, ampiezza_spike))

    if not valid_spikes:
        return None

    # Se ci sono piu' spike validi si tiene quello di ampiezza maggiore
    valid_spikes.sort(key=lambda x: x[2], reverse=True)
    best_start, best_end, _ = valid_spikes[0]
    return on + best_start, on + best_end + 1


def _minimo_dopo_R(sig, r, limite, baseline):
    """Posizione della S: minimo assoluto dopo R, entro il limite dato.

    Il limite e' qrs_off, oppure R' quando esiste un secondo picco positivo.
    Delimitare la ricerca con R' e' cio' che impedisce alla S di scivolare oltre.
    """
    r, limite = int(r), int(limite)
    if limite <= r + 1:
        return limite
    return _extremum(sig, r, limite, mode="min", baseline=baseline)


def _extremum_polarity(sig, start, end, polarity, baseline=0.0, prominence_frac=0.10,
                       override_ratio=1.1):
    """Trova un picco nel tratto [start, end] preferendo una data polarita'.

    La polarita' attesa e' una PREFERENZA superabile, non una regola rigida: se il
    miglior candidato di polarita' opposta e' almeno override_ratio volte piu' ampio di
    quello atteso, si sceglie l'opposto. Questo gestisce i casi, frequenti con QRS largo
    o asse deviato, in cui l'onda ha polarita' atipica per quella derivazione.

    Il segnale viene detrendato, cioe' privato della retta che congiunge gli estremi del
    tratto, e finestrato con un taper. In questo modo i picchi che cadono sui bordi non
    vengono premiati artificialmente dalla ricerca.
    """
    start, end = int(start), int(end)
    if end <= start:
        return start

    segment = sig[start:end + 1]
    N = len(segment)

    # Detrend: rimuove la pendenza di fondo del tratto
    x = np.arange(N)
    y_start, y_end = segment[0], segment[-1]
    line = y_start + (y_end - y_start) * x / max(N - 1, 1)
    detrended = segment - line

    # Taper ai bordi: evita che un picco sul confine domini la ricerca
    window = np.ones(N)
    taper_len = int(max(1, min(N * 0.15, 10)))
    for idx in range(taper_len):
        factor = 0.5 * (1.0 - np.cos(np.pi * idx / taper_len))
        window[idx] = factor
        window[-1 - idx] = factor
    detrended_windowed = detrended * window

    escursione = float(np.ptp(detrended_windowed)) if N > 1 else 0.0
    prom = max(escursione * prominence_frac, 0.003)

    massimi, _ = find_peaks(detrended_windowed, prominence=prom)
    minimi, _ = find_peaks(-detrended_windowed, prominence=prom)
    massimi = [start + m for m in massimi]
    minimi = [start + m for m in minimi]

    # Nessuna polarita' attesa: si prende la deviazione piu' ampia
    if polarity not in ('pos', 'neg'):
        candidati = massimi + minimi
        if candidati:
            return max(candidati, key=lambda m: abs(detrended_windowed[m - start]))
        idx_max = int(np.argmax(segment))
        idx_min = int(np.argmin(segment))
        if abs(segment[idx_max] - segment[0]) >= abs(segment[idx_min] - segment[0]):
            return start + idx_max
        return start + idx_min

    candidati_pos = [m for m in massimi if detrended_windowed[m - start] > 0.003]
    candidati_neg = [m for m in minimi if detrended_windowed[m - start] < -0.003]

    best_pos = max(candidati_pos, key=lambda m: detrended_windowed[m - start]) if candidati_pos else None
    best_neg = min(candidati_neg, key=lambda m: detrended_windowed[m - start]) if candidati_neg else None

    amp_pos = detrended_windowed[best_pos - start] if best_pos is not None else 0.0
    amp_neg = abs(detrended_windowed[best_neg - start]) if best_neg is not None else 0.0

    if polarity == 'pos':
        atteso, opposto, amp_atteso, amp_opposto = best_pos, best_neg, amp_pos, amp_neg
    else:
        atteso, opposto, amp_atteso, amp_opposto = best_neg, best_pos, amp_neg, amp_pos

    # Override: la polarita' opposta vince solo se nettamente piu' ampia
    if opposto is not None and amp_opposto >= override_ratio * max(amp_atteso, 1e-9):
        return opposto
    if atteso is not None:
        return atteso
    if opposto is not None:
        return opposto

    return start + int(np.argmax(segment)) if polarity == 'pos' else start + int(np.argmin(segment))


def _separa_qrs_degenere(sig, q, r, s, qrs_on, qrs_off):
    """Garantisce che Q, R ed S siano tre indici distinti e ordinati.

    Nei complessi QS puri non esiste nessuna deflessione positiva: il ventricolo si
    depolarizza solo verso il basso in quella derivazione. In quel caso il massimo del
    tratto cade sul primo campione, e i detector finiscono per collassare Q ed R sullo
    stesso indice (la clausola "if q > r: q = r" cristallizza la coincidenza).

    La convenzione del progetto vuole che Q, R ed S siano sempre presenti e distinti.
    Questo helper la ripristina separando gli indici, MA senza inventare un picco che non
    esiste: la R viene collocata sul punto piu' alto del breve tratto compreso fra Q ed S,
    cioe' sulla spalla iniziale del complesso.

    In particolare la R NON viene riancorata al massimo globale della finestra. In un QS
    quel massimo e' un artefatto: nel pacing sarebbe lo spike di stimolazione, in altri
    casi il bordo destro del QRS. Riancorarla la' rimetterebbe la R esattamente dove i fix
    precedenti l'avevano tolta.

    Il flag is_qs_complex, gia' calcolato dai detector, resta il modo corretto per sapere
    che in quella derivazione l'onda R e' fisiologicamente assente.
    """
    if q is None or r is None or s is None:
        return q, r, s
    q, r, s = int(q), int(r), int(s)

    # Caso normale: i tre indici sono gia' distinti e ordinati, non si tocca nulla
    if q < r < s:
        return q, r, s

    # Caso degenere: c'e' spazio per collocare R fra Q ed S.
    # Q ed S calcolati dal detector sono corretti (sono estremi veri), quindi si tengono e
    # si cerca la R come massimo del tratto interno, estremi esclusi.
    if s - q >= 2:
        interno_lo, interno_hi = q + 1, s - 1
        if interno_hi >= interno_lo:
            rel = int(np.argmax(sig[interno_lo:interno_hi + 1]))
            return q, interno_lo + rel, s

    # Tratto troppo stretto per tre indici distinti: si allarga verso i confini del QRS
    q_new = max(qrs_on, min(q, s - 2)) if s - 2 >= qrs_on else qrs_on
    r_new = q_new + 1
    s_new = max(r_new + 1, s)
    if s_new > qrs_off:
        s_new = qrs_off
        r_new = max(q_new + 1, s_new - 1)
    return q_new, r_new, s_new


def _estendi_s_in_st(sig, s, qrs_off, margine=12, esplorazione=50, freno=0.10):
    """Post-passo ST_T: recupera la S quando il sottoslivellamento la nasconde.

    Nel sottoslivellamento ST a scodella il vero fondo della S puo' cadere appena fuori
    dal confine QRS predetto dal modello, che si ferma troppo presto per via del rumore
    di linea. Se la S si e' fermata entro `margine` campioni dal bordo, si sospetta un
    falso arresto e si esplora oltre il confine in cerca di un punto piu' profondo.

    Freno di sicurezza: appena il segnale risale di piu' di `freno` mV rispetto al fondo
    trovato, la scodella e' finita e sta iniziando l'onda T, quindi la ricerca si ferma.

    Questo passo e' applicato DOPO qualunque detector QRS. In questo modo un ECG che e'
    insieme ST_T e WIDE_QRS conserva sia la morfologia del QRS largo, sia il recupero
    della S nel tratto ST: le due logiche si sommano invece di escludersi.
    """
    if s is None or qrs_off is None:
        return s
    s, qrs_off = int(s), int(qrs_off)

    # Si attiva solo se la S e' sospettosamente vicina al bordo destro del QRS
    if s < qrs_off - margine or qrs_off >= len(sig) - 1:
        return s

    limite_fuga = min(len(sig) - 1, qrs_off + esplorazione)
    s_fuga = s

    for i in range(s, limite_fuga):
        if sig[i] < sig[s_fuga]:
            s_fuga = i
        # Freno: il segnale sta risalendo, siamo usciti dalla scodella
        if sig[i] - sig[s_fuga] > freno:
            break

    return s_fuga if sig[s_fuga] < sig[s] else s


# =====================================================================
# 7. DETECTOR QRS PER MACRO-CLASSE
#
# Convenzioni valide per tutti i detector QRS:
#   - R e' sempre presente e non e' mai un punto nettamente negativo
#   - Q e' sempre presente ed e' il minimo prima di R
#   - S e' sempre presente ed e' il minimo dopo R
#   - vale sempre l'ordine temporale Q < R < S, e Q < R < S < R' quando R' esiste
#   - la baseline e' la mediana del segnale, non un valore locale, perche' e' molto piu'
#     robusta rispetto alle deflessioni patologiche di grande ampiezza
# =====================================================================

def detect_qrs_standard(sig, qrs_on, qrs_off, lead_name=None, qs_threshold_frac=0.15, qr_dominance=3.0, qr_min_frac=0.15):
    """QRS standard. Usato per HEALTHY, NO_P e UNKNOWN."""
    qrs_on, qrs_off = int(qrs_on), int(qrs_off)
    if qrs_off <= qrs_on + 2:
        return {"Q": None, "R": None, "S": None, "is_qs_complex": None}

    baseline = float(np.median(sig))
    seg = sig[qrs_on:qrs_off + 1] - baseline

    r_cand = qrs_on + int(np.argmax(seg))
    s_cand = qrs_on + int(np.argmin(seg))
    val_r = abs(sig[r_cand] - baseline)
    val_s = abs(sig[s_cand] - baseline)

    # ================= BYPASS ANTI-STEMI V1-V3 =================
    lead_upper = (lead_name or "").upper().strip()
    if lead_upper in {"V1", "V2", "V3"} and val_s > 0.05:
        s_out = int(s_cand) # Forziamo la S nel cratere
        
        # Troviamo la R iniziale (o la forziamo se non c'è)
        r_cand_pre = _extremum(sig, qrs_on, s_out, mode="max", baseline=baseline)
        r_out = r_cand_pre if (r_cand_pre is not None and sig[r_cand_pre] - baseline > 0) else qrs_on
        
        # Troviamo la Q iniziale
        q_cand_pre = _extremum(sig, qrs_on, r_out, mode="min", baseline=baseline)
        q_out = q_cand_pre if q_cand_pre is not None else qrs_on
        
        # Costringiamo Q < R < S per sicurezza matematica
        q_out = max(qrs_on, min(q_out, qrs_off - 2))
        r_out = max(q_out + 1, min(r_out, qrs_off - 1))
        s_out = max(r_out + 1, min(s_out, qrs_off))
        
        is_qs = bool(val_r < qs_threshold_frac * val_s)
        return {"Q": q_out, "R": r_out, "S": s_out, "is_qs_complex": is_qs}
    # ===========================================================

    is_qs = bool(val_s > 0 and val_r < qs_threshold_frac * val_s)
    vince_r = (val_r >= val_s)

    if vince_r:
        r = r_cand
        q_cand = _extremum(sig, qrs_on, r, mode="min", baseline=baseline)
        q = q_cand if (q_cand is not None and r > qrs_on) else qrs_on
        s_cand_2 = _extremum(sig, r, qrs_off, mode="min", baseline=baseline)
        s = s_cand_2 if (s_cand_2 is not None and r < qrs_off) else qrs_off
    else:
        profondita_valle = abs(sig[s_cand] - baseline)
        cand_prima = _extremum(sig, qrs_on, s_cand, mode="max", baseline=baseline) if s_cand > qrs_on else None
        amp_prima = max(0.0, (sig[cand_prima] - baseline)) if cand_prima is not None else 0.0
        
        if cand_prima is None and s_cand > qrs_on:
            amp_prima = max(0.0, sig[qrs_on] - baseline)
            cand_prima = qrs_on

        cand_dopo = _extremum(sig, s_cand, qrs_off, mode="max", baseline=baseline) if s_cand < qrs_off else None
        amp_dopo = max(0.0, (sig[cand_dopo] - baseline)) if cand_dopo is not None else 0.0

        is_qr_pattern = (amp_dopo >= qr_dominance * max(amp_prima, 1e-9)) and (amp_dopo >= qr_min_frac * profondita_valle)

        if not is_qr_pattern:
            s = s_cand
            r = cand_prima if (cand_prima is not None and sig[cand_prima] - baseline > 0) else qrs_on
            q_cand = _extremum(sig, qrs_on, r, mode="min", baseline=baseline) if r > qrs_on else None
            q = q_cand if q_cand is not None else qrs_on
        else:
            q = s_cand
            r = cand_dopo if cand_dopo is not None else qrs_off
            s_cand_2 = _extremum(sig, r, qrs_off, mode="min", baseline=baseline) if r < qrs_off else None
            s = s_cand_2 if s_cand_2 is not None else qrs_off

    if q is None: q = qrs_on
    if r is None: r = q + 1
    if s is None: s = r + 1

    if r <= q: r = min(q + 1, qrs_off - 1)
    if q >= r: q = max(qrs_on, r - 1)
    if s <= r: s = min(qrs_off, r + 1)

    return {"Q": q, "R": r, "S": s, "is_qs_complex": is_qs}


def detect_qrs_old_mi(sig, qrs_on, qrs_off, lead_name=None, qs_threshold_frac=0.25):
    """QRS per infarto pregresso (OLD_MI).

    Pattern atteso: onde Q profonde e patologiche, complessi QS da necrosi.

    Usa lo stesso motore di detect_qrs_standard, ma con qs_threshold_frac alzata da 0.15
    a 0.25. La soglia oltre cui un complesso e' classificato come QS diventa quindi piu'
    permissiva, cosi' i complessi con R residua ma molto ridotta, tipici della necrosi,
    vengono correttamente riconosciuti come QS.
    """
    return detect_qrs_standard(sig, qrs_on, qrs_off, lead_name=lead_name, qs_threshold_frac=qs_threshold_frac)


def detect_qrs_hypertrophy(sig, qrs_on, qrs_off, lead_name=None, qs_threshold_frac=0.15):
    """QRS per ipertrofia ventricolare (HYPERTROPHY).

    Pattern atteso: voltaggi molto elevati, morfologia per il resto conservata.

    Usa il motore standard senza modifiche ai parametri, perche' e' gia' adatto: cerca
    gli estremi assoluti in modo diretto, senza euristiche di prominenza relativa che su
    voltaggi molto alti darebbero risultati instabili.

    La funzione esiste come alias esplicito per rendere leggibile il dispatch e per
    poter differenziare i parametri in futuro senza toccare le altre classi.
    """
    return detect_qrs_standard(sig, qrs_on, qrs_off, lead_name=lead_name, qs_threshold_frac=qs_threshold_frac)


def detect_qrs_wide_qrs(sig, qrs_on, qrs_off, lead_name=None, qs_threshold_frac=0.15,
                        prominence_frac=0.05, r_prime_frac=0.30):
    """QRS largo: blocchi di branca, emiblocchi, ritardi di conduzione (WIDE_QRS)."""
    qrs_on, qrs_off = int(qrs_on), int(qrs_off)
    if qrs_off <= qrs_on + 2:
        return {"Q": None, "R": None, "S": None, "R_prime": None, "is_qs_complex": None}

    baseline = float(np.median(sig))
    seg = sig[qrs_on:qrs_off + 1] - baseline
    r_cand = qrs_on + int(np.argmax(seg))
    s_cand = qrs_on + int(np.argmin(seg))
    val_r = abs(sig[r_cand] - baseline)
    val_s = abs(sig[s_cand] - baseline)
    is_qs = bool(val_s > 0 and val_r < qs_threshold_frac * val_s)

    candidate_before = _extremum(sig, qrs_on, s_cand, mode="max", baseline=baseline) if s_cand > qrs_on else qrs_on
    noise_thresh = max(0.015, prominence_frac * (val_r + val_s))
    exists_before = candidate_before is not None and (candidate_before > qrs_on) and (sig[candidate_before] - baseline) > noise_thresh

    # ================= BYPASS ANTI-STEMI V1-V3 (WIDE QRS) =================
    lead_upper = (lead_name or "").upper().strip()
    if lead_upper in {"V1", "V2", "V3"} and val_s > 0.05:
        s_out = int(s_cand)
        r_out = candidate_before if (candidate_before is not None and sig[candidate_before] - baseline > 0) else qrs_on
        q_cand = _extremum(sig, qrs_on, r_out, mode="min", baseline=baseline)
        q_out = q_cand if q_cand is not None else qrs_on

        q_out = max(qrs_on, min(q_out, qrs_off - 2))
        r_out = max(q_out + 1, min(r_out, qrs_off - 1))
        s_out = max(r_out + 1, min(s_out, qrs_off))
        
        return {"Q": q_out, "R": r_out, "S": s_out, "R_prime": None, "is_qs_complex": is_qs}
    # ======================================================================

    if exists_before:
        r = candidate_before
        s = s_cand
        q_cand = _extremum(sig, qrs_on, r, mode="min", baseline=baseline) if r > qrs_on else None
        q = q_cand if q_cand is not None else qrs_on
    else:
        q = s_cand
        massimi_dopo, _ = _local_extrema(sig, s_cand, qrs_off, prominence_frac=prominence_frac)
        cand_dopo = [m for m in massimi_dopo if (sig[m] - baseline) > 0]
        if cand_dopo:
            r = max(cand_dopo, key=lambda m: sig[m] - baseline)
            s_cand_2 = _minimo_dopo_R(sig, r, qrs_off, baseline) if r < qrs_off else qrs_off
            s = s_cand_2 if s_cand_2 is not None else qrs_off
        else:
            r = r_cand
            if r <= q:
                q = min(q, r - 1) if r > qrs_on else qrs_on
            s_cand_2 = _minimo_dopo_R(sig, r, qrs_off, baseline) if r < qrs_off else qrs_off
            s = s_cand_2 if s_cand_2 is not None else qrs_off

    r_prime = None
    val_r_eff = abs(sig[r] - baseline)
    massimi, _ = _local_extrema(sig, s, qrs_off, prominence_frac=prominence_frac)
    massimi_dopo2 = [m for m in massimi if m > s and (sig[m] - baseline) > 0]
    if massimi_dopo2:
        cand = max(massimi_dopo2, key=lambda m: sig[m] - baseline)
        if abs(sig[cand] - baseline) >= r_prime_frac * val_r_eff:
            r_prime = cand
            if r < r_prime:
                s_cand_3 = _extremum(sig, r, r_prime, mode="min", baseline=baseline)
                s = s_cand_3 if s_cand_3 is not None else s

    if q != qrs_on and (sig[q] - baseline) >= 0:
        q = qrs_on

    if r <= q: r = min(q + 1, qrs_off - 1)
    if q >= r: q = max(qrs_on, r - 1)
    if s <= r:
        if r < qrs_off: s = r + 1
        else: r = max(q + 1, r - 1); s = qrs_off
    if r_prime is not None and r_prime <= s:
        r_prime = None

    return {"Q": q, "R": r, "S": s, "R_prime": r_prime, "is_qs_complex": is_qs}


def detect_qrs_avblock(sig, qrs_on, qrs_off, lead_name=None, qs_threshold_frac=0.15):
    """QRS per blocco atrio-ventricolare (AV_BLOCK) — con gestione spike pacemaker.

    I pazienti con blocco AV hanno spesso un pacemaker impiantato: lo spike elettrico
    del pacemaker puo' cadere dentro la finestra QRS e confondere il detector se non
    viene neutralizzato. La funzione:

      1. Cerca uno spike di pacing nei primi campioni della finestra QRS.
      2. Se lo trova e il median filter conferma che e' un artefatto (ampiezza smooth
         molto inferiore a quella raw), sposta l'inizio della ricerca subito dopo lo spike.
      3. Sulla finestra "pulita" applica la stessa logica di detect_qrs_standard, incluso
         il bypass anti-STEMI per V1-V3.
    """
    qrs_on = int(max(0, qrs_on))
    qrs_off = int(min(len(sig) - 1, qrs_off))

    if qrs_off <= qrs_on + 2:
        return {"Q": qrs_on, "R": qrs_on + 1, "S": qrs_off, "is_qs_complex": False}

    # --- 1. GESTIONE DELLO SPIKE (Backup per pacemaker) ---
    spike_search_on = max(0, qrs_on - 15)
    spike = _detect_pacing_spike(sig, spike_search_on, qrs_off, max_width_samples=4,
                                 slope_factor=5.0, return_tol_frac=0.9, min_slope=0.10,
                                 allow_biphasic=True)

    ricerca_on = qrs_on
    if spike:
        st = max(0, spike[0] - spike_search_on)
        en = min(qrs_off - spike_search_on, spike[1] - spike_search_on)
        baseline_test = float(np.median(sig[spike_search_on:qrs_off + 1]))
        seg_test = sig[spike_search_on:qrs_off + 1] - baseline_test

        k_size = 5 if len(seg_test) >= 5 else 3
        seg_smooth = medfilt(seg_test, kernel_size=k_size)

        raw_amp = np.max(np.abs(seg_test[st:en + 1]))
        smooth_amp = np.max(np.abs(seg_smooth[st:en + 1]))

        if smooth_amp < 0.6 * raw_amp:
            ricerca_on = max(qrs_on, min(qrs_off - 2, spike[1]))
            if ricerca_on + 1 < qrs_off:
                ricerca_on += 1

    # --- 2. DELEGA A detect_qrs_standard SUL SEGMENTO PULITO ---
    return detect_qrs_standard(sig, ricerca_on, qrs_off, lead_name=lead_name,
                               qs_threshold_frac=qs_threshold_frac)


def detect_qrs_pacing(sig, qrs_on, qrs_off, lead_name=None):
    """QRS in presenza di pacemaker (PACING) - Corretto per spike e ringing."""
    qrs_on = int(max(0, qrs_on))
    qrs_off = int(min(len(sig) - 1, qrs_off))

    # Fase 1: individuazione e conferma dello spike
    # Guardiamo 15 campioni indietro per la salita
    spike_search_on = max(0, qrs_on - 15)
    
    # PARAMETRI ULTRA-PERMISSIVI: abbassiamo biphasic_min_amp a 0.2
    spike = _detect_pacing_spike(sig, spike_search_on, qrs_off, 
                                 max_width_samples=5, slope_factor=3.0, 
                                 return_tol_frac=1.5, min_slope=0.05,
                                 allow_biphasic=True, biphasic_max_width=10, biphasic_min_amp=0.2)

    spike_info = {"spike_start": spike[0], "spike_end": spike[1]} if spike else None
    ricerca_on = qrs_on

    if spike:
        st = max(0, spike[0] - spike_search_on)
        en = min(qrs_off - spike_search_on, spike[1] - spike_search_on)
        baseline_test = float(np.median(sig[spike_search_on:qrs_off + 1]))
        seg_test = sig[spike_search_on:qrs_off + 1] - baseline_test

        k_size = 5 if len(seg_test) >= 5 else 3
        seg_smooth = medfilt(seg_test, kernel_size=k_size)

        raw_amp = np.max(np.abs(seg_test[st:en + 1]))
        smooth_amp = np.max(np.abs(seg_smooth[st:en + 1]))

        # Conferma: se l'ampiezza svanisce col lisciamento, era davvero uno spike
        if smooth_amp < 0.8 * raw_amp:
            # SALTIAMO LO SPIKE + 3 CAMPIONI PER EVITARE IL CRATERE
            ricerca_on = max(qrs_on, min(qrs_off - 2, spike[1] + 3))

    if qrs_off <= ricerca_on + 2:
        return {"Q": ricerca_on, "R": ricerca_on + 1, "S": qrs_off,
                "R_prime": None, "is_qs_complex": False, "pacing_spike": spike_info}

    # Fase 2: analisi morfologica del tratto ripulito dallo spike
    baseline = _baseline(sig, ricerca_on)
    seg_norm = sig[ricerca_on:qrs_off + 1] - baseline

    idx_max = int(np.argmax(seg_norm))
    idx_min = int(np.argmin(seg_norm))
    val_max = seg_norm[idx_max]
    val_min = abs(seg_norm[idx_min])

    r_prime = None
    vince_s = bool(val_min > val_max * 1.5)

    if vince_s:
        # Cratere dominante: pattern QS o rS
        s = ricerca_on + idx_min
        if idx_min > 0:
            r = ricerca_on + int(np.argmax(seg_norm[:idx_min]))
        else:
            r = s
        if r > ricerca_on:
            q = ricerca_on + int(np.argmin(seg_norm[:int(r - ricerca_on)]))
        else:
            q = r
        is_qs = bool(val_max < 0.15 * val_min)

    else:
        if idx_min < idx_max:
            # Il minimo precede il massimo
            seg_before_min = seg_norm[:idx_min]

            if len(seg_before_min) > 0 and np.max(seg_before_min) > 0.03:
                # Pattern rSR': c'e' un'ondina positiva prima della valle
                r_rel = int(np.argmax(seg_before_min))
                r = ricerca_on + r_rel
                s = ricerca_on + idx_min
                r_prime = ricerca_on + idx_max
                if r_rel > 0:
                    q = ricerca_on + int(np.argmin(seg_norm[:r_rel]))
                else:
                    q = r
            else:
                # Pattern qR: nessuna ondina iniziale, la valle e' una vera onda Q
                q = ricerca_on + idx_min
                r = ricerca_on + idx_max
                if idx_max < len(seg_norm) - 1:
                    s = r + 1 + int(np.argmin(seg_norm[idx_max + 1:]))
                else:
                    s = r
        else:
            # Pattern Rs classico: il massimo precede il minimo
            r = ricerca_on + idx_max
            s = ricerca_on + idx_min
            if idx_max > 0:
                q = ricerca_on + int(np.argmin(seg_norm[:idx_max]))
            else:
                q = r

        is_qs = False

    # Vincoli di dominio e ordinamento
    q = max(ricerca_on, min(q, qrs_off))
    r = max(ricerca_on, min(r, qrs_off))
    s = max(ricerca_on, min(s, qrs_off))

    if q > r:
        q = r
    if s < r:
        s = r

    # La Q non puo' essere clinicamente piu' alta della R
    if sig[q] > sig[r]:
        q = r

    # Separazione minima tra i marker, ma solo se non altera la morfologia
    if q == r and r > ricerca_on and sig[r - 1] <= sig[r]:
        q = r - 1
    if s == r and r < qrs_off and sig[r + 1] <= sig[r]:
        s = r + 1

    return {"Q": q, "R": r, "S": s, "R_prime": r_prime,
            "is_qs_complex": is_qs, "pacing_spike": spike_info}


# =====================================================================
# 8. DETECTOR P, T, J
# =====================================================================

def detect_p_standard(sig, p_on, p_off, lead_name=None):
    """Onda P fisiologica: picco con la polarita' attesa per quella derivazione."""
    baseline = _baseline(sig, p_on)
    lead_upper = (lead_name or "").upper()
    polarity = P_POLARITY_PER_LEAD.get(lead_upper)

    # In V1, III e AVL la P e' spesso piatta: si alza la soglia di prominenza
    prom_frac = 0.15 if lead_upper in P_FIRST_FALLBACK_LEADS else 0.10

    p = _extremum_polarity(sig, p_on, p_off, polarity, baseline=baseline,
                           prominence_frac=prom_frac)
    return {"P": p}


def detect_p_avblock(sig, p_on, p_off, lead_name=None):
    """Onda P nel blocco AV.

    Il blocco AV rallenta o interrompe la conduzione dall'atrio al ventricolo, ma
    l'impulso nasce comunque dal nodo del seno: la polarita' della P resta quella
    sinusale. L'unica differenza rispetto al detector standard e' quindi la mappa di
    polarita' usata, che conferma esplicitamente la positivita' in II, III e AVF.
    """
    baseline = _baseline(sig, p_on)
    lead_upper = (lead_name or "").upper()
    polarity = P_POLARITY_AVBLOCK.get(lead_upper)

    prom_frac = 0.15 if lead_upper in P_FIRST_FALLBACK_LEADS else 0.10

    p = _extremum_polarity(sig, p_on, p_off, polarity, baseline=baseline,
                           prominence_frac=prom_frac)
    return {"P": p}


def detect_p_no_p(sig, p_on, p_off, lead_name=None, min_width_samples=15, noise_factor=2.5):
    """Onda P assente: fibrillazione atriale, flutter, ritmi giunzionali (NO_P).

    In queste aritmie l'onda P non esiste. Il detector deve quindi poter rispondere None,
    invece di marcare per forza un picco sul rumore. Due controlli in cascata:
      1. Se il tratto P predetto e' troppo stretto, non c'e' abbastanza segnale su cui
         cercare qualcosa di significativo.
      2. Se l'escursione del tratto e' paragonabile al rumore di fondo del segnale, cio'
         che si vedrebbe sarebbe rumore, non un'onda.

    Se entrambi i controlli passano, il picco viene marcato ma segnalato con il flag
    P_da_verificare, perche' potrebbe comunque essere un'onda f di fibrillazione e non
    una vera onda P.
    """
    on, off = int(p_on), int(p_off)
    if off - on < min_width_samples:
        return {"P": None}

    escursione = float(np.ptp(sig[on:off + 1]))
    if escursione < noise_factor * float(np.std(sig)):
        return {"P": None}

    risultato = detect_p_standard(sig, p_on, p_off, lead_name=lead_name)
    risultato["P_da_verificare"] = True
    return risultato


def _detect_p_biphasic(sig, p_on, p_off, p_principale, baseline,
                       min_amp_frac=0.20, min_amp_abs=0.015):
    """Individua la seconda fase di un'onda P bifasica.

    Una P bifasica ha due deflessioni di segno opposto e ravvicinate. In V1 la componente
    negativa terminale, se ampia e prolungata, indica dilatazione atriale sinistra.
    Si cerca quindi un estremo di segno opposto al picco principale, entro una distanza
    massima e con ampiezza non trascurabile.

    Ritorna la tripla (p_prima, p_seconda, is_bifasica), con i due indici in ordine
    temporale.
    """
    if p_principale is None or p_on is None or p_off is None:
        return p_principale, None, False

    massimi, minimi = _local_extrema(sig, p_on, p_off, prominence_frac=0.10)
    candidati = sorted(set(massimi + minimi + [p_principale]))

    secondario = None
    amp_principale = abs(sig[p_principale] - baseline)

    for cand in candidati:
        if cand == p_principale:
            continue

        stesso_segno = np.sign(sig[cand] - baseline) == np.sign(sig[p_principale] - baseline)
        gap = abs(cand - p_principale)
        amp_cand = abs(sig[cand] - baseline)

        if not stesso_segno and gap <= P_BIPHASIC_GAP_MAX_SAMPLES:
            if amp_cand >= min_amp_frac * amp_principale and amp_cand >= min_amp_abs:
                secondario = cand
                break

    if secondario is None:
        return p_principale, None, False

    p_prima, p_seconda = sorted([p_principale, secondario])
    return p_prima, p_seconda, True


def detect_t_standard(sig, t_on, t_off, lead_name=None):
    """Onda T fisiologica: picco con la polarita' attesa per quella derivazione."""
    baseline = _baseline(sig, t_on)
    polarity = T_POLARITY_PER_LEAD.get((lead_name or "").upper())
    return {"T": _extremum_polarity(sig, t_on, t_off, polarity, baseline=baseline)}


def detect_t_stt(sig, t_on, t_off, lead_name=None, prominence_frac=0.15):
    """Onda T alterata: ischemia, sovraccarico, ripolarizzazione anomala (ST_T).

    In queste condizioni la T e' spesso invertita o bifasica. A differenza del detector
    standard, qui si lavora sui massimi e minimi locali prominenti e si cerca
    esplicitamente una seconda componente di segno opposto.
    """
    baseline = _baseline(sig, t_on)
    massimi, minimi = _local_extrema(sig, t_on, t_off, prominence_frac=prominence_frac)
    candidati = massimi + minimi
    if not candidati:
        return {"T": None}

    polarity = T_POLARITY_PER_LEAD.get((lead_name or "").upper())
    if polarity == 'pos':
        candidati_segno = [c for c in candidati if sig[c] - baseline > 0]
    elif polarity == 'neg':
        candidati_segno = [c for c in candidati if sig[c] - baseline < 0]
    else:
        candidati_segno = []

    # 1. Trova il picco più ampio in ASSOLUTO e quello col SEGNO ATTESO
    best_overall = max(candidati, key=lambda i: abs(sig[i] - baseline))
    best_expected = max(candidati_segno, key=lambda i: abs(sig[i] - baseline)) if candidati_segno else None

    # 2. OVERRIDE: Se la polarita' opposta vince in modo schiacciante
    # la T è chiaramente invertita e ignoriamo il micro-candidato atteso.
    if best_expected is not None:
        # ABBASSATO DA 2.0 a 1.1 per permettere alla T invertita di vincere
        if abs(sig[best_overall] - baseline) > 1.1 * abs(sig[best_expected] - baseline):
            t_principale = best_overall
        else:
            t_principale = best_expected
    else:
        t_principale = best_overall

    # Seconda componente di segno opposto: T bifasica
    t_secondario = None
    for cand in sorted(candidati, key=lambda i: -abs(sig[i] - baseline)):
        if cand == t_principale:
            continue
        stesso_segno = np.sign(sig[cand] - baseline) == np.sign(sig[t_principale] - baseline)
        if not stesso_segno and abs(sig[cand] - baseline) >= 0.40 * abs(sig[t_principale] - baseline):
            t_secondario = cand
            break

    return {"T": t_principale, "T_bifasica_secondaria": t_secondario}


def detect_j_point(sig, s_idx, qrs_off, t_on, lead_name=""):
    """Punto J: esteso fino al 15% del tratto ST per catturare l'onda J senza invadere la T."""
    # (Nota: se avevi commentato il blocco J_POINT_LEADS per avere la J 
    # su tutte le derivazioni, lascialo commentato anche qui)
    
    if s_idx is None or qrs_off is None:
        return None

    s_idx = int(s_idx)
    qrs_off = int(qrs_off)
    
    # 1. Calcoliamo il 15% del tratto ST (tra fine QRS e inizio T)
    if t_on is not None and t_on > qrs_off:
        limite_st = qrs_off + int((t_on - qrs_off) * 0.15)
    else:
        # Fallback di sicurezza se la rete non ha trovato la T
        limite_st = qrs_off + 15 

    # 2. Vincolo: la ricerca inizia rigorosamente DOPO la S
    inizio_j = min(s_idx + 1, len(sig) - 1)
    fine = min(limite_st, len(sig) - 1)

    if fine <= inizio_j:
        return inizio_j

    segmento = sig[inizio_j:fine + 1] - _baseline(sig, s_idx)
    
    # 3. Cerca i picchi locali (J-wave) con una minima prominenza
    peaks, _ = find_peaks(segmento, prominence=0.02)
    
    if len(peaks) > 0:
        return inizio_j + peaks[0]  # Prende il primo picco positivo dopo S
    else:
        # Fallback: torna al massimo assoluto entro il 15% del tratto ST
        return inizio_j + int(np.argmax(segmento))


def measure_st_deviation(sig, j_point, qrs_on):
    """Deviazione del punto J dalla baseline, in mV. Positiva = sopraslivellamento."""
    if j_point is None or qrs_on is None:
        return None
    baseline = _baseline(sig, qrs_on)
    return float(sig[int(j_point)] - baseline)


# =====================================================================
# 9. DISPATCH DEI DETECTOR (GESTIONE MULTI-PATOLOGIA)
#
# Un ECG reale porta spesso piu' diagnosi contemporaneamente, per esempio blocco di branca
# sinistra insieme a ipertrofia ventricolare e ad alterazioni ST aspecifiche. Assegnare al
# file una sola macro-classe e ignorare le altre significa perdere informazione: un caso
# con blocco di branca E fibrillazione atriale verrebbe trattato solo come WIDE_QRS, e il
# codice continuerebbe a cercare un'onda P che non esiste.
#
# L'osservazione chiave e' che le patologie NON competono per lo stesso picco:
#   PACING, WIDE_QRS, AV_BLOCK, OLD_MI, HYPERTROPHY  agiscono sul complesso QRS
#   NO_P                                             agisce sull'onda P
#   ST_T                                             agisce sull'onda T
#
# Ogni onda puo' quindi scegliere il proprio detector in modo indipendente dalle altre, e
# le diagnosi coesistono invece di escludersi a vicenda.
#
# L'unica eccezione e' il recupero della S nel sottoslivellamento ST, che tocca il QRS pur
# essendo una caratteristica ST_T. E' implementato come post-passo (_estendi_s_in_st)
# applicato dopo qualunque detector QRS, cosi' un ECG che e' insieme ST_T e WIDE_QRS
# conserva entrambe le logiche.
# =====================================================================

# Gerarchia dei detector QRS: si sceglie il primo che compare tra le classi attive.
# L'ordine riflette quanto profondamente la patologia altera la morfologia del complesso:
# lo spike del pacemaker va rimosso prima di ogni altra analisi, poi vengono le alterazioni
# della conduzione, poi quelle del tessuto, infine il solo aumento di voltaggio.
QRS_PRIORITY = ['PACING', 'WIDE_QRS', 'AV_BLOCK', 'OLD_MI', 'HYPERTROPHY']

QRS_DETECTORS = {
    'PACING': detect_qrs_pacing,
    'WIDE_QRS': detect_qrs_wide_qrs,
    'AV_BLOCK': detect_qrs_avblock,
    'OLD_MI': detect_qrs_old_mi,
    'HYPERTROPHY': detect_qrs_hypertrophy,
    'HEALTHY': detect_qrs_standard,
}


def select_detectors(matched):
    """Sceglie i detector di QRS, P e T a partire dall'insieme delle classi attive.

    Ogni onda decide in modo indipendente, secondo la propria gerarchia. Il valore di
    ritorno include anche i nomi dei detector scelti, che vengono poi salvati nel CSV:
    cosi' per ogni picco si sa esattamente quale logica lo ha prodotto, ed e' possibile
    stratificare i risultati per patologia anche sui casi con diagnosi multiple.

    Ritorna (qrs_func, p_func, t_func), (qrs_label, p_label, t_label), applica_fuga_st
    """
    # QRS: prima classe attiva secondo QRS_PRIORITY, altrimenti il detector standard
    qrs_label = 'HEALTHY'
    for c in QRS_PRIORITY:
        if c in matched:
            qrs_label = c
            break
    qrs_func = QRS_DETECTORS[qrs_label]

    # P: solo NO_P e AV_BLOCK hanno una logica dedicata
    if 'NO_P' in matched:
        p_label, p_func = 'NO_P', detect_p_no_p
    elif 'AV_BLOCK' in matched:
        p_label, p_func = 'AV_BLOCK', detect_p_avblock
    else:
        p_label, p_func = 'HEALTHY', detect_p_standard

    # T: solo ST_T ha una logica dedicata
    if 'ST_T' in matched:
        t_label, t_func = 'ST_T', detect_t_stt
    else:
        t_label, t_func = 'HEALTHY', detect_t_standard

    # Il recupero della S nel tratto ST si applica ogni volta che ST_T e' attiva,
    # qualunque sia il detector QRS effettivamente scelto
    applica_fuga_st = 'ST_T' in matched

    return (qrs_func, p_func, t_func), (qrs_label, p_label, t_label), applica_fuga_st


def find_all_peaks(sig, boundaries, matched, lead_name, fs=TARGET_FS):
    """Trova tutti i picchi di una derivazione: P, Q, R, S, T, J.

    matched e' l'insieme delle macro-classi attive per questo ECG. I detector sono scelti
    da select_detectors(), che tratta ogni onda in modo indipendente dalle altre.

    Oltre ai picchi vengono calcolati alcuni marcatori clinici derivati: ampiezza e
    morfologia della P (bifasica, bifida, pulmonale), Q patologica, deviazione del
    tratto ST.
    """
    sig = np.asarray(sig, dtype=float)
    lead_upper = lead_name.upper()

    (qrs_func, p_func, t_func), (qrs_label, p_label, t_label), applica_fuga_st = select_detectors(matched)

    risultato = {
        "detector_QRS": qrs_label,
        "detector_P": p_label,
        "detector_T": t_label,
    }

    # Onda P
    p_on_input = boundaries["P_Onset"]
    if p_on_input is not None and boundaries["P_Offset"] is not None:

        # Spike atriale: nel pacing atriale precede l'onda P e va escluso dalla ricerca
        spike_search_on = max(0, p_on_input - 5)
        if 'PACING' in matched:
            spike_atriale = _detect_pacing_spike(sig, spike_search_on, boundaries["P_Offset"],
                                                 max_width_samples=5, slope_factor=4.0,
                                                 return_tol_frac=1.0, min_slope=0.01)
        else:
            spike_atriale = _detect_pacing_spike(sig, spike_search_on, boundaries["P_Offset"],
                                                 max_width_samples=4, slope_factor=5.0,
                                                 return_tol_frac=0.9, min_slope=0.12)

        if spike_atriale is not None:
            spike_start, spike_end = spike_atriale
            if spike_end < boundaries["P_Offset"]:
                p_on_input = min(spike_end + 8, boundaries["P_Offset"])
                risultato["pacing_spike_atriale"] = {"spike_start": spike_start,
                                                     "spike_end": spike_end}

        risultato.update(p_func(sig, p_on_input, boundaries["P_Offset"], lead_name=lead_name))

        # P bifasica: cercata solo in II e V1, dove e' clinicamente significativa
        if lead_upper in P_BIFASICA_LEADS and risultato.get("P") is not None:
            baseline_p_bif = _baseline(sig, boundaries["P_Onset"])
            p_prima, p_seconda, is_bifasica = _detect_p_biphasic(
                sig, p_on_input, boundaries["P_Offset"], risultato["P"], baseline_p_bif
            )
            risultato["P_prime"] = p_prima
            risultato["P_second"] = p_seconda
            risultato["is_p_bifasica"] = is_bifasica

            # P terminal force in V1: durata per ampiezza della componente negativa
            # terminale. Sopra 40 ms per mm indica dilatazione atriale sinistra.
            risultato["P_terminal_force_anomala"] = False
            if is_bifasica and lead_upper == "V1" and p_seconda is not None:
                ampiezza_negativa = baseline_p_bif - sig[p_seconda]
                durata_terminale_ms = (boundaries["P_Offset"] - p_seconda) * 1000.0 / fs
                ptfv1 = durata_terminale_ms * (ampiezza_negativa * 10.0)
                if ptfv1 >= 40.0:
                    risultato["P_terminal_force_anomala"] = True
        else:
            risultato["P_prime"] = risultato.get("P")
            risultato["P_second"] = None
            risultato["is_p_bifasica"] = False
            risultato["P_terminal_force_anomala"] = False

    # Complesso QRS
    qrs_on_input = boundaries["QRS_Onset"]
    spike_universale = None

    # Fuori dal PACING lo spike puo' comunque essere presente, per esempio se il pacemaker
    # non e' dichiarato nel referto: si cerca sempre, con parametri conservativi.
    # Nel PACING la rimozione e' gia' dentro detect_qrs_pacing, che usa anche la verifica
    # col filtro mediano, quindi qui non si interviene per non rimuoverlo due volte.
   # === ELIMINA O COMMENTA QUESTO BLOCCO ===
    # if 'PACING' not in matched and qrs_on_input is not None and boundaries["QRS_Offset"] is not None:
    #     spike_universale = _detect_pacing_spike(sig, qrs_on_input, boundaries["QRS_Offset"],
    #                                             max_width_samples=4, slope_factor=5.0,
    #                                             return_tol_frac=0.9, min_slope=0.12)
    #     if spike_universale is not None:
    #         spike_start, spike_end = spike_universale
    #         if spike_end < boundaries["QRS_Offset"]:
    #             qrs_on_input = spike_end
    # ========================================

    if boundaries["QRS_Onset"] is not None and boundaries["QRS_Offset"] is not None:
        # Passa lead_name al detector QRS
        risultato.update(qrs_func(sig, qrs_on_input, boundaries["QRS_Offset"], lead_name=lead_name))

        # Separazione dei picchi degeneri. Nei complessi QS puri i detector collassano Q
        # ed R sullo stesso indice, perche' non esiste nessun massimo positivo. Qui si
        # ripristina la convenzione Q < R < S senza spostare la R su un massimo spurio.
        risultato["Q"], risultato["R"], risultato["S"] = _separa_qrs_degenere(
            sig, risultato.get("Q"), risultato.get("R"), risultato.get("S"),
            qrs_on_input, boundaries["QRS_Offset"]
        )

        # Post-passo ST_T: recupera la S se il sottoslivellamento la nasconde oltre il
        # confine del QRS. Applicato dopo qualunque detector, cosi' si combina con
        # WIDE_QRS, AV_BLOCK e gli altri invece di escluderli.
        # === ELIMINA O COMMENTA QUESTO BLOCCO ===
        # Post-passo ST_T: recupera la S se il sottoslivellamento la nasconde oltre il
        # confine del QRS. Applicato dopo qualunque detector, cosi' si combina con
        # WIDE_QRS, AV_BLOCK e gli altri invece di escluderli.
        # if applica_fuga_st and risultato.get("S") is not None:
        #     risultato["S"] = _estendi_s_in_st(sig, risultato["S"], boundaries["QRS_Offset"])
        # ========================================

        if spike_universale is not None and "pacing_spike" not in risultato:
            risultato["pacing_spike"] = {"spike_start": spike_universale[0],
                                         "spike_end": spike_universale[1]}
    else:
        risultato.update({"Q": None, "R": None, "S": None})

    # Onda T
    if boundaries["T_Onset"] is not None and boundaries["T_Offset"] is not None:
        risultato.update(t_func(sig, boundaries["T_Onset"], boundaries["T_Offset"],
                                lead_name=lead_name))
    else:
        risultato["T"] = None

    # Punto J e deviazione del tratto ST
    inizio_ricerca_j = risultato.get("S") if risultato.get("S") is not None else risultato.get("R")
    
    # Passiamo sia il QRS_Offset che il T_Onset per calcolare la metà dell'ST
    risultato["J"] = detect_j_point(sig, inizio_ricerca_j, 
                                    boundaries.get("QRS_Offset"), 
                                    boundaries.get("T_Onset"), 
                                    lead_name)
                                    
    risultato["ST_deviation"] = measure_st_deviation(sig, risultato["J"], boundaries.get("QRS_Onset"))

    # Marcatori clinici derivati

    # Ampiezza della P: la soglia e' piu' bassa dove la P pulmonale e' visibile
    if risultato.get("P") is not None and boundaries.get("P_Onset") is not None:
        baseline_p = _baseline(sig, boundaries["P_Onset"])
        ampiezza_p = abs(sig[risultato["P"]] - baseline_p)
        if lead_upper in P_PULMONALE_LEADS:
            soglia_p = 0.20
        elif lead_upper in PRECORDIALI:
            soglia_p = 0.15
        else:
            soglia_p = 0.25
        risultato["P_amplitude_mV"] = float(ampiezza_p)
        risultato["P_amplitude_anomala"] = bool(ampiezza_p > soglia_p)
    else:
        risultato["P_amplitude_mV"] = None
        risultato["P_amplitude_anomala"] = None

    # Q patologica: in V1, V2 e V3 una Q apprezzabile e distinta dalla R e' anomala
    if lead_upper in Q_ANOMALA_LEADS and risultato.get("Q") is not None and boundaries.get("QRS_Onset") is not None:
        baseline_qrs = _baseline(sig, boundaries["QRS_Onset"])
        ampiezza_q = abs(sig[risultato["Q"]] - baseline_qrs)
        risultato["Q_anomala"] = bool(ampiezza_q > 0.02 and risultato["Q"] != risultato.get("R"))
    else:
        risultato["Q_anomala"] = False

    # P bifida: due cuspidi positive distanziate tra 40 e 100 ms, con la seconda non
    # trascurabile. E' un segno di dilatazione atriale sinistra.
    risultato["P_bifida"] = False
    if lead_upper in P_BIFIDA_LEADS and risultato.get("P") is not None and boundaries.get("P_Onset") is not None:
        baseline_p2 = _baseline(sig, boundaries["P_Onset"])
        massimi_p, _ = _local_extrema(sig, boundaries["P_Onset"], boundaries["P_Offset"],
                                      prominence_frac=0.10)
        massimi_pos_p = sorted([m for m in massimi_p if sig[m] - baseline_p2 > 0])

        if len(massimi_pos_p) >= 2:
            primo, secondo = massimi_pos_p[0], massimi_pos_p[1]
            gap_ms = (secondo - primo) * 1000.0 / fs
            ampiezza_primo = sig[primo] - baseline_p2
            ampiezza_secondo = sig[secondo] - baseline_p2

            if 40 <= gap_ms <= 100 and ampiezza_secondo >= 0.5 * ampiezza_primo:
                risultato["P_bifida"] = True

    return risultato


# =====================================================================
# 10. PLOTTING
# =====================================================================

PEAK_STYLE = {
    "P": dict(marker="v", color="#1565C0", label="P"),
    "Q": dict(marker="x", color="#6A1B9A", label="Q"),
    "R": dict(marker="*", color="#B71C1C", label="R"),
    "S": dict(marker="x", color="#6A1B9A", label="S"),
    "T": dict(marker="^", color="#2E7D32", label="T"),
    "J": dict(marker="D", color="#EF6C00", label="J"),
}


def _mark_peak(ax, sig, t_ms, idx, key, fs=TARGET_FS, fontsize=7, label_override=None, show_text=True):
    # Disegna un singolo marker di picco con la sua etichetta
    if idx is None:
        return
    style = PEAK_STYLE[key]
    x = t_ms[idx]
    y = sig[idx]
    size = 90 if key == "R" else 40
    kwargs = dict(marker=style["marker"], color=style["color"], s=size, zorder=8)
    if style["marker"] != "x":
        kwargs.update(edgecolors="white", linewidths=0.5)
    ax.scatter([x], [y], **kwargs)
    
    if show_text:
        etichetta = label_override if label_override is not None else style["label"]
        ax.annotate(etichetta, (x, y), textcoords="offset points",
                    xytext=(3, 4), fontsize=fontsize, color=style["color"],
                    fontweight="bold", zorder=9)


def shade(ax, a, b, color, alpha, label=None):
    # Colora l'intervallo [a, b] sull'asse
    if a is not None and b is not None and b > a:
        ax.axvspan(a, b, alpha=alpha, color=color, label=label, zorder=2)


def plot_12_leads_with_peaks(signals_filt, gt_mask, pred_mask, matched, macro_class,
                             fname, output_path, fs=TARGET_FS):
    """Confronto a due colonne: ground truth GE a sinistra, predizione con picchi a destra.

    I boundary sono condivisi da tutte le derivazioni, perche' il modello lavora sulla
    media delle probabilita' delle 12 derivazioni, come nella convenzione GE. I picchi,
    invece, sono cercati derivazione per derivazione.
    """
    t_ms = np.arange(MEDIAN_LEN) * 1000 / fs

    gt_ivs_samples = extract_intervals(gt_mask, fs)
    pred_ivs = extract_intervals(pred_mask, fs)
    gt_ivs_ms = {k: (v * 1000 / fs if v is not None else None) for k, v in gt_ivs_samples.items()}
    pred_ivs_ms = {k: (v * 1000 / fs if v is not None else None) for k, v in pred_ivs.items()}

    qrs_off_ms = pred_ivs_ms.get('QRS_Offset')
    t_on_ms = pred_ivs_ms.get('T_Onset')
    has_st = (qrs_off_ms is not None and t_on_ms is not None and t_on_ms > qrs_off_ms)

    boundaries_samples = dict(pred_ivs)
    boundaries_samples["ST_Onset"] = pred_ivs.get("QRS_Offset")
    boundaries_samples["ST_Offset"] = pred_ivs.get("T_Onset")

    # Si disegnano solo le derivazioni realmente presenti: quelle piatte sono state
    # scartate a monte e non devono comparire come tracce vuote con picchi spuri
    leads_validi = [l for l in LEADS_ORDER if l in signals_filt]
    n_righe = max(1, len(leads_validi))

    fig, axes = plt.subplots(n_righe, 2, figsize=(18, 38 * n_righe / 12), sharex=True)
    if n_righe == 1:
        axes = np.array([axes])

    # Nel titolo si riportano tutte le classi attive, non solo quella principale
    classi_str = ' + '.join(sorted(matched)) if matched else macro_class
    titolo_extra = ''
    if len(leads_validi) < len(LEADS_ORDER):
        scartate = sorted(set(LEADS_ORDER) - set(leads_validi))
        titolo_extra = f'\nderivazioni piatte scartate: {", ".join(scartate)}'
    fig.suptitle(
        f'CONFRONTO {len(leads_validi)} DERIVAZIONI (con picchi) - {fname}\n'
        f'classe principale: {macro_class}   |   classi attive: {classi_str}{titolo_extra}',
        fontsize=15, fontweight='bold', y=0.996
    )

    peaks_per_lead = {}

    for i, lead in enumerate(leads_validi):
        sig = signals_filt[lead]

        # Colonna sinistra: ground truth GE
        ax0 = axes[i, 0]
        ax0.plot(t_ms, sig, color='#212121', lw=0.8)
        shade(ax0, gt_ivs_ms.get('P_Onset'), gt_ivs_ms.get('P_Offset'), C_P, 0.25)
        shade(ax0, gt_ivs_ms.get('QRS_Onset'), gt_ivs_ms.get('QRS_Offset'), C_QRS, 0.25)
        shade(ax0, gt_ivs_ms.get('T_Onset'), gt_ivs_ms.get('T_Offset'), C_TGT, 0.25)
        ax0.set_ylabel(f'{lead}', fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        ax0.grid(True, alpha=0.15)
        if i == 0:
            ax0.set_title('GE Ground Truth', fontsize=14, pad=10)

        # Colonna destra: predizione del modello con i picchi marcati
        ax1 = axes[i, 1]
        ax1.plot(t_ms, sig, color='#212121', lw=0.8)
        shade(ax1, pred_ivs_ms.get('P_Onset'), pred_ivs_ms.get('P_Offset'), C_P, 0.25)
        shade(ax1, pred_ivs_ms.get('QRS_Onset'), pred_ivs_ms.get('QRS_Offset'), C_QRS, 0.25)
        if has_st:
            shade(ax1, qrs_off_ms, t_on_ms, C_ST, 0.45)
        shade(ax1, pred_ivs_ms.get('T_Onset'), pred_ivs_ms.get('T_Offset'), C_T, 0.35)
        ax1.grid(True, alpha=0.15)
        if i == 0:
            ax1.set_title('Predizione Ensemble + picchi rilevati', fontsize=14, pad=10)

        peaks = find_all_peaks(sig, boundaries_samples, matched, lead, fs=fs)
        peaks_per_lead[lead] = peaks

        for key in ("Q", "R", "S", "T"):
            _mark_peak(ax1, sig, t_ms, peaks.get(key), key, fs=fs)
        _mark_peak(ax1, sig, t_ms, peaks.get("J"), "J", fs=fs)

        # Con P bifasica si marca la prima delle due componenti
        if peaks.get("is_p_bifasica"):
            _mark_peak(ax1, sig, t_ms, peaks.get("P_prime"), "P", fs=fs)
        else:
            _mark_peak(ax1, sig, t_ms, peaks.get("P"), "P", fs=fs)

        # R' viene calcolata e salvata nel CSV, ma non disegnata: nei complessi rSR' il
        # plot diventerebbe illeggibile e la triade Q-R-S e' l'informazione principale

        # Seconda componente della T bifasica
        if peaks.get("T_bifasica_secondaria") is not None:
            idx = peaks["T_bifasica_secondaria"]
            ax1.scatter([t_ms[idx]], [sig[idx]], marker="^", color="#81C784",
                        s=30, zorder=8, edgecolors="white", linewidths=0.4)
            ax1.annotate("T2", (t_ms[idx], sig[idx]), textcoords="offset points",
                         xytext=(3, 4), fontsize=6.5, color="#81C784", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=120)
    plt.close()

    return peaks_per_lead, boundaries_samples


# =====================================================================
# 11. PIPELINE PRINCIPALE
# =====================================================================

def carica_modelli(model_v3_path, model_fase6_path, device):
    """Carica i due modelli dell'ensemble.

    Fase2 (EnhancedUNetV3) fornisce le maschere di P e QRS.
    Fase6 (ECGUNetFase6) fornisce la maschera di T: e' il modello addestrato per
    distillazione, che corregge la convenzione GE secondo cui T_Onset = QRS_Offset.
    """
    model_v3 = EnhancedUNetV3(in_channels=1, num_classes=4, f=12).to(device)
    model_v3.load_state_dict(torch.load(model_v3_path, map_location=device, weights_only=True))
    model_v3.eval()

    model_f6 = ECGUNetFase6(in_channels=1, num_classes=4).to(device)
    ckpt_f6 = torch.load(model_fase6_path, map_location=device, weights_only=False)
    if isinstance(ckpt_f6, dict) and 'model_state_dict' in ckpt_f6:
        ckpt_f6 = ckpt_f6['model_state_dict']

    # Il checkpoint puo' avere le chiavi prefissate con "backbone."
    cleaned_f6 = {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k: v
                  for k, v in ckpt_f6.items()}
    model_f6.load_state_dict(cleaned_f6, strict=False)
    model_f6.eval()

    return model_v3, model_f6


def predici_maschera(signals, model_v3, model_f6, device):
    """Applica l'ensemble a un ECG e restituisce la maschera segmentata.

    I due modelli sono applicati a ogni derivazione separatamente, poi le probabilita'
    sono mediate sulle 12 derivazioni: la segmentazione e' quindi un unico set di boundary
    condiviso, coerente con la convenzione GE del battito mediano.

    La composizione e' class-wise: l'onda T viene dalla Fase6, la P e il QRS dalla Fase2.

    Si iterano solo le derivazioni realmente presenti in signals. Quelle piatte sono gia'
    state scartate da parse_xml, e vanno lasciate fuori: rigenerarle come array di zeri le
    farebbe rientrare dalla finestra, inquinando la media delle probabilita' e producendo
    picchi spuri sui primi campioni.
    """
    lead_scores_v3, lead_scores_f6, signals_filt = [], [], {}
    leads_validi = [l for l in LEADS_ORDER if l in signals]

    with torch.no_grad():
        for lead in leads_validi:
            raw_sig = signals[lead]
            filt_sig = apply_ecg_filters(raw_sig, TARGET_FS)
            signals_filt[lead] = filt_sig

            sig_norm = normalize(filt_sig)
            x = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            lead_scores_v3.append(torch.softmax(model_v3(x), dim=1).squeeze(0))
            lead_scores_f6.append(torch.softmax(model_f6(x), dim=1).squeeze(0))

    # Media delle probabilita' sulle 12 derivazioni
    pred_v3_glob = torch.stack(lead_scores_v3).mean(0).argmax(0).cpu().numpy()
    pred_f6_glob = torch.stack(lead_scores_f6).mean(0).argmax(0).cpu().numpy()

    # Composizione class-wise. L'ordine di scrittura conta: P e QRS sono scritti dopo la T
    # e quindi hanno la precedenza in caso di sovrapposizione, perche' i loro confini sono
    # piu' affidabili di quelli della T.
    pred_mask = np.zeros(MEDIAN_LEN, dtype=np.int64)
    pred_mask[pred_f6_glob == 3] = 3
    pred_mask[pred_v3_glob == 1] = 1
    pred_mask[pred_v3_glob == 2] = 2

    return postprocess_mask(pred_mask), signals_filt


def raccogli_xml(xml_root_dir, only_class=None, only_file=None, limit=None):
    """Raccoglie i file XML da processare.

    Accetta due organizzazioni della cartella:
      a) sottocartelle per macro-classe, per esempio XML patologie/WIDE_QRS/*.xml
      b) cartella piatta di XML, senza sottocartelle

    In entrambi i casi la classe usata per il peak detection e' quella letta dagli
    statement del referto, non quella dedotta dal nome della cartella: la cartella e' solo
    un modo di organizzare i file sul disco.
    """
    sottocartelle = sorted([d for d in os.listdir(xml_root_dir)
                            if os.path.isdir(os.path.join(xml_root_dir, d))])

    xml_files = []

    if sottocartelle:
        # Organizzazione a): sottocartelle per macro-classe
        if only_class is not None:
            only_class = only_class.upper()
            if only_class not in sottocartelle:
                print(f"ATTENZIONE: la cartella '{only_class}' non esiste in {xml_root_dir}.")
                print(f"Cartelle disponibili: {sottocartelle}")
                return []
            sottocartelle = [only_class]

        for d in sottocartelle:
            xml_files.extend(sorted(glob.glob(os.path.join(xml_root_dir, d, "*.xml"))))
    else:
        # Organizzazione b): cartella piatta
        xml_files = sorted(glob.glob(os.path.join(xml_root_dir, "*.xml")))

    if only_file is not None:
        xml_files = [f for f in xml_files if only_file in os.path.basename(f)]
    if limit is not None:
        xml_files = xml_files[:limit]

    return xml_files


def run(xml_root_dir, model_v3_path, model_fase6_path, output_dir,
        only_class=None, limit=None, only_file=None):
    """Pipeline completa: parsing, segmentazione, peak detection, plot e CSV.

    La macro-classe di ogni file e' determinata dal codice a partire dagli statement
    diagnostici contenuti nell'XML, non dalla cartella in cui il file si trova. Il
    parametro only_class filtra comunque per cartella quando la struttura e' organizzata
    per classe, ed e' utile per i test mirati su una singola patologia.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_v3, model_f6 = carica_modelli(model_v3_path, model_fase6_path, device)
    print("Modelli Ensemble caricati con successo.")

    os.makedirs(output_dir, exist_ok=True)

    xml_files = raccogli_xml(xml_root_dir, only_class=only_class,
                             only_file=only_file, limit=limit)
    print(f"File XML da processare: {len(xml_files)}")

    righe_csv = []
    conteggio_classi = {}
    conteggio_combinazioni = {}
    conteggio_mancanti = {k: 0 for k in ("P", "Q", "R", "S", "T", "J")}
    totale_leads_processate = 0
    file_scartati = []

    for xml_path in xml_files:
        fname = os.path.basename(xml_path)

        parsed = parse_xml(xml_path)
        if parsed is None:
            # Il record e' illeggibile oppure il battito mediano e' vuoto (tutte le
            # derivazioni piatte). Va escluso dalle statistiche: non e' un fallimento
            # dell'algoritmo di peak detection, e' un dato assente all'origine.
            print(f"  [SCARTATO] {fname} - formato non riconosciuto o battito mediano vuoto")
            file_scartati.append(fname)
            continue
        signals, global_ann, statements = parsed

        # Se mancano derivazioni (perche' piatte), si processano solo quelle valide
        leads_validi = [l for l in LEADS_ORDER if l in signals]
        if len(leads_validi) < len(LEADS_ORDER):
            mancanti = set(LEADS_ORDER) - set(leads_validi)
            print(f"  [ATTENZIONE] {fname} - derivazioni piatte scartate: {sorted(mancanti)}")

        # La classe viene dedotta dal referto, non dalla cartella
        matched = get_matched_classes(statements)
        macro_class = get_macro_class(statements)
        if not matched:
            matched = {'HEALTHY'}

        conteggio_classi[macro_class] = conteggio_classi.get(macro_class, 0) + 1
        combo = ' + '.join(sorted(matched))
        conteggio_combinazioni[combo] = conteggio_combinazioni.get(combo, 0) + 1

        # Segmentazione con l'ensemble
        pred_mask, signals_filt = predici_maschera(signals, model_v3, model_f6, device)
        gt_mask = build_gt_mask(global_ann, fs=TARGET_FS) if global_ann else np.zeros(MEDIAN_LEN, dtype=np.int64)

        # Il plot viene salvato nella cartella della classe principale
        dest_dir = os.path.join(output_dir, macro_class)
        os.makedirs(dest_dir, exist_ok=True)
        img_path = os.path.join(dest_dir, os.path.splitext(fname)[0] + '_picchi.png')

        peaks_per_lead, boundaries_condivisi = plot_12_leads_with_peaks(
            signals_filt, gt_mask, pred_mask, matched, macro_class, fname, img_path, fs=TARGET_FS
        )
        print(f"  [OK] {fname} -> {macro_class} ({combo})")

        for lead, peaks in peaks_per_lead.items():
            totale_leads_processate += 1
            for k in ("P", "Q", "R", "S", "T", "J"):
                if peaks.get(k) is None:
                    conteggio_mancanti[k] += 1

            righe_csv.append({
                "file": fname,
                # Classe principale e insieme completo delle diagnosi attive
                "macro_class": macro_class,
                "macro_classes_all": '|'.join(sorted(matched)),
                # Detector effettivamente usato per ciascuna onda: permette di stratificare
                # i risultati anche sui casi con diagnosi multiple
                "detector_QRS": peaks.get("detector_QRS"),
                "detector_P": peaks.get("detector_P"),
                "detector_T": peaks.get("detector_T"),
                "lead": lead,
                "P_idx": peaks.get("P"), "Q_idx": peaks.get("Q"), "R_idx": peaks.get("R"),
                "S_idx": peaks.get("S"), "T_idx": peaks.get("T"), "J_idx": peaks.get("J"),
                "R_prime_idx": peaks.get("R_prime"),
                "ST_deviation_mV": peaks.get("ST_deviation"),
                "pacing_spike": peaks.get("pacing_spike"),
                "pacing_spike_atriale": peaks.get("pacing_spike_atriale"),
                "T_bifasica_secondaria_idx": peaks.get("T_bifasica_secondaria"),
                "is_qs_complex": peaks.get("is_qs_complex"),
                "P_da_verificare": peaks.get("P_da_verificare"),
                "P_amplitude_mV": peaks.get("P_amplitude_mV"),
                "P_amplitude_anomala": peaks.get("P_amplitude_anomala"),
                "Q_anomala": peaks.get("Q_anomala"),
                "P_bifida": peaks.get("P_bifida"),
                "P_prime_idx": peaks.get("P_prime"),
                "P_second_idx": peaks.get("P_second"),
                "is_p_bifasica": peaks.get("is_p_bifasica"),
                "P_terminal_force_anomala": peaks.get("P_terminal_force_anomala"),
                "boundary_QRS_Onset": boundaries_condivisi.get("QRS_Onset"),
                "boundary_QRS_Offset": boundaries_condivisi.get("QRS_Offset"),
                "boundary_P_Onset": boundaries_condivisi.get("P_Onset"),
                "boundary_P_Offset": boundaries_condivisi.get("P_Offset"),
                "boundary_T_Onset": boundaries_condivisi.get("T_Onset"),
                "boundary_T_Offset": boundaries_condivisi.get("T_Offset"),
            })

    df = pd.DataFrame(righe_csv)
    csv_path = os.path.join(output_dir, "picchi_dettaglio.csv")
    df.to_csv(csv_path, index=False)

    # Riepilogo finale
    print(f"\n{'=' * 60}")
    print("RIEPILOGO")
    print(f"{'=' * 60}")

    print(f"\nFile processati: {len(xml_files) - len(file_scartati)} su {len(xml_files)}")
    if file_scartati:
        print(f"File scartati (battito mediano vuoto o formato non valido): {len(file_scartati)}")
        for f in file_scartati:
            print(f"  {f}")

    print("\nFile per classe principale:")
    for classe, n in sorted(conteggio_classi.items(), key=lambda x: -x[1]):
        print(f"  {classe}: {n}")

    print("\nCombinazioni di diagnosi riscontrate:")
    for combo, n in sorted(conteggio_combinazioni.items(), key=lambda x: -x[1]):
        print(f"  {combo}: {n}")

    print(f"\nDerivazioni processate in totale: {totale_leads_processate}")
    print("Picchi non trovati (None) per tipo:")
    for k, n in conteggio_mancanti.items():
        pct = 100 * n / totale_leads_processate if totale_leads_processate else 0
        print(f"  {k}: {n} ({pct:.1f}%)")

    print(f"\nCSV di dettaglio: {csv_path}")
    print(f"Plot salvati in: {output_dir}/<MACRO_CLASSE>/")


def main():
    parser = argparse.ArgumentParser(description="Trova i picchi P/Q/R/S/T/J con l'Ensemble2+6")
    parser.add_argument("--xml-root-dir", default=DEFAULT_XML_ROOT_DIR,
                        help="Cartella con gli XML: puo' avere sottocartelle per classe o essere piatta")
    parser.add_argument("--model-v3", default=os.path.join(DEFAULT_MODEL_DIR, "best_fase2_v3.pt"))
    parser.add_argument("--model-fase6", default=os.path.join(DEFAULT_MODEL_DIR, "best_fase6.pt"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-class", default=None,
                        help="Processa solo questa sottocartella, per esempio WIDE_QRS, utile per test mirati")
    parser.add_argument("--only-file", default=None,
                        help="Processa solo i file il cui nome contiene questa stringa")
    parser.add_argument("--limit", type=int, default=None,
                        help="Processa solo i primi N file")

    # parse_known_args e' necessario in Colab: il kernel Jupyter passa argomenti propri che
    # parse_args interpreterebbe come errore
    args, _ = parser.parse_known_args()

    run(args.xml_root_dir, args.model_v3, args.model_fase6, args.output_dir,
        only_class=args.only_class, limit=args.limit, only_file=args.only_file)


# if __name__ == "__main__":
#     main()