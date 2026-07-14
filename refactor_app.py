import re

with open(r'c:\Users\maria\Desktop\TESI_DRIVE\APP\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add outward_search_boundaries
outward_func = """
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

"""
if "def outward_search_boundaries" not in content:
    content = content.replace("def build_median_gt_mask", outward_func + "\ndef build_median_gt_mask")


# 2. Update build_median_gt_mask to NOT use fixed pacing logic, just use the passed annotations
# And build_continuous_mask_from_ann to do the same.
# Let's replace the whole block of those two functions.
import re

mask_funcs_regex = re.compile(r'def build_median_gt_mask.*?def stamp_median_mask', re.DOTALL)
mask_funcs_new = """def build_median_gt_mask(global_ann, fs=TARGET_FS):
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
    if t_onset is not None and t_offset is not None:
        s, e = to_idx(t_onset), to_idx(t_offset)
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

def stamp_median_mask"""

content = mask_funcs_regex.sub(mask_funcs_new, content)

# 3. Add 12-lead plot function
plot_12_leads_func = """
def plot_12_leads_comparison(median_sigs, global_ann, models, matched, fs=TARGET_FS):
    model_v3, model_f6, device = models
    fig, axes = plt.subplots(12, 2, figsize=(16, 2.5 * 12), sharex=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle("CONFRONTO 12 DERIVAZIONI (con picchi)", fontsize=16, fontweight='bold', y=0.92)
    
    gt_mask_base = build_median_gt_mask(global_ann, fs)
    t_ms = np.arange(MEDIAN_LEN) * 1000 / fs
    
    for i, lead in enumerate(LEADS_ORDER):
        ax_gt = axes[i, 0]
        ax_pred = axes[i, 1]
        
        if i == 0:
            ax_gt.set_title("GT Ground Truth")
            ax_pred.set_title("Predizione Ensemble + Picchi")
            
        ax_gt.set_ylabel(lead, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        
        if lead not in median_sigs:
            ax_gt.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_gt.transAxes)
            ax_pred.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_pred.transAxes)
            ax_gt.set_yticks([]); ax_pred.set_yticks([])
            continue
            
        sig = median_sigs[lead]
        
        # 1. Inferenza
        sig_norm = ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float32)
        sig_pad = np.pad(sig_norm, (4, 4), mode='constant')
        x = torch.tensor(sig_pad).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            s_v3 = torch.nn.functional.softmax(model_v3(x), dim=1).squeeze(0).cpu()
            s_f6 = torch.nn.functional.softmax(model_f6(x), dim=1).squeeze(0).cpu()
        
        pv3 = s_v3[:, 4:604].argmax(0).numpy()
        pf6 = s_f6[:, 4:604].argmax(0).numpy()
        
        pred_mask = np.zeros(MEDIAN_LEN, dtype=np.int64)
        pred_mask[pf6 == 3] = 3
        pred_mask[pv3 == 1] = 1
        pred_mask[pv3 == 2] = 2
        pred_mask = postprocess_mask(pred_mask, fs=fs)
        
        # 2. Trova picchi
        # find_all_peaks assume il segnale grezzo/filtrato ma in millivolt per l'ampiezza
        sig_for_peaks = apply_ecg_filters(sig, fs=fs)
        peaks_dict, _ = find_all_peaks(sig_for_peaks, pred_mask, matched)
        
        # Plot GT
        ax_gt.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
        shade_segs(ax_gt, extract_intervals_multibeat(gt_mask_base), show_st=False)
        ax_gt.grid(True, alpha=0.3)
        
        # Plot Pred
        ax_pred.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
        shade_segs(ax_pred, extract_intervals_multibeat(pred_mask), show_st=True)
        ax_pred.grid(True, alpha=0.3)
        
        # Mark peaks
        for p_name, (pos_x, pos_y) in peaks_dict.items():
            if p_name == 'P': _mark_peak(ax_pred, pos_x, pos_y, 'P', 'blue')
            elif p_name in ['Q', 'R', 'S']: _mark_peak(ax_pred, pos_x, pos_y, p_name, 'red')
            elif p_name == 'T': _mark_peak(ax_pred, pos_x, pos_y, 'T', 'green')
            elif p_name == 'J': _mark_peak(ax_pred, pos_x, pos_y, 'J', 'orange')
            
    # Solo ultima riga ha etichette x
    for ax in axes.flatten():
        if ax not in axes[-1,:]: ax.set_xticklabels([])
    
    return fig

"""
if "def plot_12_leads_comparison" not in content:
    content = content.replace("def generate_plot", plot_12_leads_func + "\ndef generate_plot")

# 4. Refactor the main layout inside main()
import textwrap

main_refactor_regex = re.compile(r'cont_container = st\.container\(\).*?if __name__ == "__main__":', re.DOTALL)
main_refactor_new = """
        global_ann_corrected = global_ann.copy() if global_ann else {}
        if is_ge:
            global_ann_corrected = get_corrected_pacing_ann(global_ann, median_sigs, matched)

        # Spostiamo il tracciato continuo in alto, forzandolo sulla Lead II se disponibile
        cont_container = st.container()
        st.markdown("---")
        med_container = st.container()

        cont_container.markdown("### Tracciato Completo (10s)")
        # Selezioniamo Lead II di default per il tracciato continuo
        lead_ii_key = next((k for k in cont_sigs.keys() if k.lower() == 'ii'), None)
        cont_lead = lead_ii_key if lead_ii_key else selected_lead
        cont_raw = cont_sigs.get(cont_lead)
        
        if cont_raw is not None:
            n_len = len(cont_raw)
            if is_ge:
                # Usa lead_scores medi o specifici per fare il timbro?
                # Il timbro si basa su pred_post del battito mediano.
                # Calcoliamo prima la predizione del battito mediano per la lead selezionata (cont_lead)
                sig_med_cont = median_sigs.get(cont_lead)
                if sig_med_cont is not None:
                    s3, s6 = run_inference(sig_med_cont, is_cont=False)
                    mask_med_pre = get_mask(s3, s6, MEDIAN_LEN)
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
                
                fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                         "Tracciato Completo", cont_lead, show_peaks, matched)
                cont_container.pyplot(fig_cont)
                
                if 'PACING' in matched:
                    cont_container.success(
                        "⚡ **Pacing Rilevato**: Il Ground Truth GE originale contiene solo lo spike. "
                        "È stata applicata la logica algoritmica automatica per rilevare i veri boundary "
                        "del complesso paceato direttamente dal segnale mediano (Outward Search)."
                    )
            else:
                # Logica LUDB (continua come prima)
                lead_scores_v3, lead_scores_f6 = {}, {}
                for l, s_ in cont_sigs.items():
                    s3, s6 = run_inference(s_, is_cont=True)
                    lead_scores_v3[l] = s3
                    lead_scores_f6[l] = s6
                
                sv3_single, sf6_single = lead_scores_v3[cont_lead], lead_scores_f6[cont_lead]
                mask_single_pre = get_mask(sv3_single, sf6_single, n_len)
                mask_single_post = postprocess_mask_multibeat(mask_single_pre, fs=TARGET_FS)
                
                pred_mask = mask_single_pre
                pred_post = mask_single_post
                gt_mask = ludb_gt_masks[cont_lead]
                
                max_samples = min(n_len, 5000)
                sig_plot = cont_raw[:max_samples]
                gt_mask_plot = gt_mask[:max_samples]
                pred_mask_plot = pred_mask[:max_samples]
                pred_post_plot = pred_post[:max_samples]
                
                fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                         "Tracciato Completo", cont_lead, show_peaks, matched)
                cont_container.pyplot(fig_cont)

        if is_ge and median_sigs:
            med_container.markdown("### Analisi 12 Derivazioni (Battito Medio)")
            
            # Calcolo metriche globali (usando le derivazioni disponibili)
            lead_scores_v3, lead_scores_f6 = {}, {}
            for l, sig in median_sigs.items():
                s3, s6 = run_inference(sig, is_cont=False)
                lead_scores_v3[l] = s3
                lead_scores_f6[l] = s6
                
            sv3_avg = torch.stack(list(lead_scores_v3.values()), dim=0).mean(dim=0)
            sf6_avg = torch.stack(list(lead_scores_f6.values()), dim=0).mean(dim=0)
            mask_avg_pre = get_mask(sv3_avg, sf6_avg, MEDIAN_LEN)
            mask_avg_post = postprocess_mask(mask_avg_pre, fs=TARGET_FS)
            
            # 12 Lead Plot
            fig_12 = plot_12_leads_comparison(median_sigs, global_ann_corrected, (model_v3, model_f6, device), matched, fs=TARGET_FS)
            med_container.pyplot(fig_12)
            
            med_container.markdown(f"#### Metriche sul Battito Medio (Tolleranza {tol_ms}ms) - Media Ensemble")
            gt_med_mask = build_median_gt_mask(global_ann_corrected, TARGET_FS)
            masks_dict = {
                "Avg12 - Pre-PP (Fase 2)": mask_avg_pre,
                "Avg12 - Post-PP (Fase 6)": mask_avg_post,
            }
            if lead_ii_key:
                mask_ii_pre = get_mask(lead_scores_v3[lead_ii_key], lead_scores_f6[lead_ii_key], MEDIAN_LEN)
                masks_dict["Lead II - Post-PP"] = postprocess_mask(mask_ii_pre, fs=TARGET_FS)
                
            render_metrics_tables(gt_med_mask, masks_dict, tol_ms, container=med_container)

if __name__ == "__main__":
"""
content = main_refactor_regex.sub(main_refactor_new, content)

with open(r'c:\Users\maria\Desktop\TESI_DRIVE\APP\app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("app.py updated successfully.")
