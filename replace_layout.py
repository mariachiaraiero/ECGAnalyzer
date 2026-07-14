import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Replace `plot_12_leads_comparison` with `plot_single_median_beat`
plot_single = """def plot_single_median_beat(lead, sig, gt_mask_base, models, matched, fs=TARGET_FS):
    model_v3, model_f6, device = models
    fig, axes = plt.subplots(1, 2, figsize=(16, 2.5), sharex=True, gridspec_kw={'wspace': 0.1})
    ax_gt, ax_pred = axes[0], axes[1]
    
    ax_gt.set_title(f"GT Ground Truth - {lead}")
    ax_pred.set_title(f"Predizione Ensemble + Picchi - {lead}")
    ax_gt.set_ylabel(lead, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
    
    t_ms = np.arange(MEDIAN_LEN) * 1000 / fs
    
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
    from picchi_final import extract_intervals
    sig_for_peaks = apply_ecg_filters(sig, fs=fs)
    boundaries = extract_intervals(pred_mask, fs=fs)
    peaks_dict = find_all_peaks(sig_for_peaks, boundaries, matched, lead_name=lead, fs=fs)
    
    # Plot GT
    ax_gt.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
    shade_segs(ax_gt, extract_intervals_multibeat(gt_mask_base), show_st=False)
    
    # Plot Pred
    ax_pred.plot(t_ms, sig, color='k', lw=1, ls='--', dash_capstyle='round', dashes=(1, 2))
    shade_segs(ax_pred, extract_intervals_multibeat(pred_mask), show_st=True)
    
    # Mark peaks
    for p_name in ['P', 'Q', 'R', 'S', 'T', 'J']:
        pos_x = peaks_dict.get(p_name)
        if pos_x is not None:
            _mark_peak(ax_pred, sig, t_ms, pos_x, p_name, fs=fs)
            
    # Formattazione
    for ax in [ax_gt, ax_pred]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    
    return fig
"""

# Sostituzione di plot_12_leads_comparison (che va da def plot_12_leads_comparison fino al suo return)
import re
plot_12_regex = re.compile(r"def plot_12_leads_comparison.*?return fig", re.DOTALL)
code = plot_12_regex.sub(plot_single, code)


# 2. Replace the main plotting logic in main()
old_main_logic = re.compile(r"cont_container = st\.container\(\).*?if __name__ == \"__main__\":", re.DOTALL)

new_main_logic = """st.markdown("---")
        
        if not is_ge:
            st.markdown("### Tracciato Completo LUDB (10s)")
            # Pre-calcolo inferenza per LUDB (che usa tutto l'elenco)
            lead_scores_v3, lead_scores_f6 = {}, {}
            for l, s_ in cont_sigs.items():
                s3, s6 = run_inference(s_, is_cont=True)
                lead_scores_v3[l] = s3
                lead_scores_f6[l] = s6

            for cont_lead, cont_raw in cont_sigs.items():
                if cont_raw is None: continue
                n_len = len(cont_raw)
                
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
                                         f"Tracciato Completo", cont_lead, show_peaks, matched)
                st.pyplot(fig_cont)
                
        else:
            st.markdown("### Analisi per Derivazione (10s + Battito Medio)")
            gt_mask_base = build_median_gt_mask(global_ann_corrected, TARGET_FS)
            
            for lead in LEADS_ORDER:
                if lead not in cont_sigs and lead not in median_sigs:
                    continue
                    
                # 1. Tracciato Continuo (se disponibile per questa derivazione)
                if lead in cont_sigs:
                    cont_raw = cont_sigs[lead]
                    n_len = len(cont_raw)
                    sig_med_cont = median_sigs.get(lead)
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
                                             f"Tracciato Completo", lead, show_peaks, matched)
                    st.pyplot(fig_cont)
                    
                # 2. Battito Mediano (se disponibile)
                if lead in median_sigs:
                    sig = median_sigs[lead]
                    fig_med = plot_single_median_beat(lead, sig, gt_mask_base, (model_v3, model_f6, device), matched, fs=TARGET_FS)
                    st.pyplot(fig_med)
                    
            if 'PACING' in matched:
                st.success(
                    "⚡ **Pacing Rilevato**: Il Ground Truth GE originale contiene solo lo spike. "
                    "È stata applicata la logica algoritmica automatica per rilevare i veri boundary "
                    "del complesso paceato direttamente dal segnale mediano (Outward Search)."
                )

            # Metriche globali battito medio
            if median_sigs:
                st.markdown(f"#### Metriche sul Battito Medio (Tolleranza {tol_ms}ms) - Media Ensemble")
                lead_scores_v3, lead_scores_f6 = {}, {}
                for l, sig in median_sigs.items():
                    s3, s6 = run_inference(sig, is_cont=False)
                    lead_scores_v3[l] = s3
                    lead_scores_f6[l] = s6
                    
                sv3_avg = torch.stack(list(lead_scores_v3.values()), dim=0).mean(dim=0)
                sf6_avg = torch.stack(list(lead_scores_f6.values()), dim=0).mean(dim=0)
                mask_avg_pre = get_mask(sv3_avg, sf6_avg, MEDIAN_LEN)
                mask_avg_post = postprocess_mask(mask_avg_pre, fs=TARGET_FS)
                
                gt_med_mask = build_median_gt_mask(global_ann_corrected, TARGET_FS)
                masks_dict = {
                    "Avg12 - Pre-PP (Fase 2)": mask_avg_pre,
                    "Avg12 - Post-PP (Fase 6)": mask_avg_post,
                }
                
                # Cerca derivazione II per metriche specifiche se disponibile
                lead_ii_key = next((k for k in median_sigs.keys() if k.lower() == 'ii'), None)
                if lead_ii_key:
                    mask_ii_pre = get_mask(lead_scores_v3[lead_ii_key], lead_scores_f6[lead_ii_key], MEDIAN_LEN)
                    masks_dict["Lead II - Post-PP"] = postprocess_mask(mask_ii_pre, fs=TARGET_FS)
                    
                render_metrics_tables(gt_med_mask, masks_dict, tol_ms, container=st)

if __name__ == "__main__":
"""

code = old_main_logic.sub(new_main_logic, code)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Sostituzione completata!")
