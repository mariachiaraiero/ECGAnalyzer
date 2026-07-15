"""
ensemble_P2_QRS5_T6.py
========================
P da Fase2, QRS da Fase5, T da Fase6 - separazione ST reale (T_Onset
distillato verso la convenzione LUDB, diverso da QRS_Offset).

Tre modelli distinti (nessuno condiviso tra le onde), quindi 3 forward pass
per derivazione invece di 2 come nella variante T2.

USO:
    python ensemble_P2_QRS5_T6.py \
        --model-fase2 /path/best_v3.pt \
        --model-fase5 /path/best_fase5.pt \
        --model-fase6 /path/best_fase6.pt \
        --xml-dir /path/XML_ECGs \
        --output-dir risultati_P2_QRS5_T6

Per restringere al test set ufficiale (evita data leakage):
    python ensemble_P2_QRS5_T6.py \
        --model-fase2 ... --model-fase5 ... --model-fase6 ... --xml-dir ... \
        --restrict-to-test-set /path/PRIV_medio/median_12dev_dataset.pt \
        --output-dir risultati_P2_QRS5_T6
"""

import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from ecg_common import (
    LEADS_ORDER, MEDIAN_LEN, TARGET_FS,
    parse_xml, build_mask, compute_aami_f1,
    apply_ecg_filters, normalize, apply_guard_window, postprocess_mask,
)


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // r), nn.ReLU(), nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(x.mean(-1)).view(b, c, 1)


class AttBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 9, padding=4), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(out_ch, out_ch, 9, padding=4), nn.BatchNorm1d(out_ch), SEBlock(out_ch))
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class MultiScaleInput(nn.Module):
    def __init__(self, in_ch=1, out_ch=12):
        super().__init__()
        c = out_ch // 3
        self.s1 = nn.Sequential(nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU())
        self.s2 = nn.Sequential(nn.AvgPool1d(2), nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU())
        self.s3 = nn.Sequential(nn.AvgPool1d(4), nn.Conv1d(in_ch, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU())

    def forward(self, x):
        L = x.size(2)
        return torch.cat([self.s1(x), F.interpolate(self.s2(x), size=L), F.interpolate(self.s3(x), size=L)], 1)


class EnhancedUNetV3(nn.Module):
    def __init__(self, f=12, dropout=0.15):
        super().__init__()
        self.ms = MultiScaleInput(1, f)
        self.enc1 = AttBlock(f, f, dropout); self.enc2 = AttBlock(f, f * 2, dropout)
        self.enc3 = AttBlock(f * 2, f * 4, dropout); self.enc4 = AttBlock(f * 4, f * 8, dropout)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = AttBlock(f * 8, f * 16, dropout)
        self.up = nn.ModuleList([nn.ConvTranspose1d(f * i, f * i, 8, 2, 3) for i in [16, 8, 4, 2]])
        self.dec = nn.ModuleList([AttBlock(f * 24, f * 8, dropout), AttBlock(f * 12, f * 4, dropout),
                                   AttBlock(f * 6, f * 2, dropout), AttBlock(f * 3, f, dropout)])
        self.final = nn.Conv1d(f, 4, 1)
        self.ds4 = nn.Conv1d(f * 8, 4, 1); self.ds3 = nn.Conv1d(f * 4, 4, 1)

    def forward(self, x):
        L = x.size(2); ms = self.ms(x)
        e1 = self.enc1(ms); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        def pm(xu, xs): return F.interpolate(xu, size=xs.size(2))

        d4 = self.dec[0](torch.cat([pm(self.up[0](b), e4), e4], 1))
        d3 = self.dec[1](torch.cat([pm(self.up[1](d4), e3), e3], 1))
        d2 = self.dec[2](torch.cat([pm(self.up[2](d3), e2), e2], 1))
        d1 = self.dec[3](torch.cat([pm(self.up[3](d2), e1), e1], 1))
        return self.final(d1)


def load_model(path, device, base_filters=12):
    model = EnhancedUNetV3(f=base_filters).to(device)
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    status = "[OK match esatto]" if not missing and not unexpected else f"[!] missing={len(missing)} unexpected={len(unexpected)}"
    print(f"   -> {status}  ({path})")
    model.eval()
    return model


def compute_scores(model, signals, device, guard_samples):
    scores = []
    with torch.no_grad():
        for lead in LEADS_ORDER:
            raw_sig = signals.get(lead, np.zeros(MEDIAN_LEN, dtype=np.float32))
            filt_sig = apply_ecg_filters(raw_sig, TARGET_FS)
            sig_norm = normalize(filt_sig)
            x = torch.tensor(sig_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            scores.append(torch.softmax(model(x), dim=1).squeeze(0))
    return apply_guard_window(torch.stack(scores).mean(0), guard_samples)


def predict_ensemble(model_p, model_qrs, model_t, signals, device, guard_samples=30):
    sc_p = compute_scores(model_p, signals, device, guard_samples)
    sc_qrs = compute_scores(model_qrs, signals, device, guard_samples)
    sc_t = compute_scores(model_t, signals, device, guard_samples)

    pred_p = sc_p.argmax(0).cpu().numpy()
    pred_qrs = sc_qrs.argmax(0).cpu().numpy()
    pred_t = sc_t.argmax(0).cpu().numpy()

    pred = np.zeros(MEDIAN_LEN, dtype=np.int64)
    pred[pred_t == 3] = 3
    pred[pred_p == 1] = 1
    pred[pred_qrs == 2] = 2
    return postprocess_mask(pred)


def get_official_test_xml_files(dataset_path, xml_dir):
    data = torch.load(dataset_path, weights_only=False)
    if 'record_metadata' not in data:
        raise ValueError(f"'record_metadata' non presente in {dataset_path}")
    record_ids = data['record_ids']
    test_idx = data['test_indices']
    test_record_ids = sorted(set(record_ids[i] for i in test_idx))
    meta = data['record_metadata']
    xml_paths = []
    for rid in test_record_ids:
        if rid in meta and 'xml_file' in meta[rid]:
            full_path = os.path.join(xml_dir, meta[rid]['xml_file'])
            if os.path.isfile(full_path):
                xml_paths.append(full_path)
    print(f" [TEST SET UFFICIALE] {len(test_record_ids)} record -> {len(xml_paths)} file XML trovati su disco")
    return sorted(xml_paths)


def eval_all(model_p, model_qrs, model_t, xml_files, device, guard_samples, output_dir, save_every, resume):
    rows = []
    partial_path = os.path.join(output_dir, "P2_QRS5_T6_xml_partial.csv")

    if resume and os.path.isfile(partial_path):
        df_partial = pd.read_csv(partial_path)
        rows = df_partial.to_dict('records')
        done_files = set(df_partial['file'])
        print(f" [RESUME] Trovati {len(done_files)} file gia' processati - li salto.")
        xml_files = [f for f in xml_files if os.path.basename(f) not in done_files]

    for xml_path in xml_files:
        fname = os.path.basename(xml_path)
        parsed = parse_xml(xml_path)
        if parsed is None:
            print(f" [SKIP] {fname} - parsing fallito"); continue
        signals, ann = parsed
        if 'Q_Onset' not in ann:
            print(f" [SKIP] {fname} - manca Q_Onset"); continue
        gt_mask = build_mask(ann)
        pred = predict_ensemble(model_p, model_qrs, model_t, signals, device, guard_samples)
        f1 = compute_aami_f1(pred, gt_mask)
        rows.append({'file': fname, 'f1_macro': f1['macro'], 'f1_P': f1['P'], 'f1_QRS': f1['QRS'], 'f1_T': f1['T']})

        n = len(rows)
        if n % 10 == 0 or n <= 3:
            print(f" [{n:4d} totali] {fname}")
        if n % save_every == 0:
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(f" [CHECKPOINT] salvati parziali a {n} file totali")

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows) if rows else None


def main():
    p = argparse.ArgumentParser(description="Ensemble P=Fase2, QRS=Fase5, T=Fase6 (separazione ST) su file XML GE MAC2000")
    p.add_argument("--model-fase2", required=True)
    p.add_argument("--model-fase5", required=True)
    p.add_argument("--model-fase6", required=True)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--output-dir", default="risultati_P2_QRS5_T6")
    p.add_argument("--restrict-to-test-set", default=None)
    p.add_argument("--base-filters", type=int, default=12)
    p.add_argument("--guard-ms", type=float, default=60.0)
    p.add_argument("--xml-sample-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guard_samples = int(round(args.guard_ms * 500 / 1000))
    print(f" Device: {device} | guard: {args.guard_ms}ms ({guard_samples} campioni)")

    for name, path in [("Fase2", args.model_fase2), ("Fase5", args.model_fase5), ("Fase6", args.model_fase6)]:
        if not os.path.isfile(path):
            print(f" [ERRORE] Checkpoint {name} non trovato: {path}"); return

    print("\nCaricamento modelli...")
    model2 = load_model(args.model_fase2, device, args.base_filters)
    model5 = load_model(args.model_fase5, device, args.base_filters)
    model6 = load_model(args.model_fase6, device, args.base_filters)

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    print(f"\n Trovati {len(xml_files)} file XML totali in {args.xml_dir}")

    if args.restrict_to_test_set:
        if not os.path.isfile(args.restrict_to_test_set):
            print(f" [ERRORE] Dataset non trovato: {args.restrict_to_test_set}"); return
        xml_files = get_official_test_xml_files(args.restrict_to_test_set, args.xml_dir)

    if args.xml_sample_n:
        rng = random.Random(args.seed)
        xml_files = rng.sample(xml_files, min(args.xml_sample_n, len(xml_files)))
        print(f" Campione casuale (seed={args.seed}): {len(xml_files)} file")

    print(f"\n{'-' * 65}\n VALUTAZIONE - {len(xml_files)} file XML - P=Fase2 QRS=Fase5 T=Fase6\n{'-' * 65}")
    df = eval_all(model2, model5, model6, xml_files, device, guard_samples,
                  args.output_dir, args.save_every, resume=not args.no_resume)

    if df is None:
        print(" Nessun file valutato con successo."); return

    csv_path = os.path.join(args.output_dir, "P2_QRS5_T6_xml.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 65}")
    print(f" P=Fase2, QRS=Fase5, T=Fase6 (separazione ST) - {len(df)} file")
    print(f"{'=' * 65}")
    print(f" F1 Macro: {df['f1_macro'].mean():.2f}% (std {df['f1_macro'].std():.2f}%)")
    print(f" F1 P:     {df['f1_P'].mean():.2f}%")
    print(f" F1 QRS:   {df['f1_QRS'].mean():.2f}%")
    print(f" F1 T:     {df['f1_T'].mean():.2f}%")
    print(f"\n [SAVED] {os.path.abspath(csv_path)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
