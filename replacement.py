        selected_lead = st.sidebar.selectbox("Seleziona Derivazione", lead_names, index=lead_names.index('I') if 'I' in lead_names else 0)
        st.title("Visualizzatore Ensemble ECG")

        def run_inference(sig, is_cont=False):
            sig_norm = ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float32)
            if is_cont:
                n_len = len(sig)
                pad_len = (16 - (n_len % 16)) % 16
                sig_pad = np.pad(sig_norm, (0, pad_len), mode='constant')
            else:
                sig_pad = np.pad(sig_norm, (4, 4), mode='constant')
            x = torch.tensor(sig_pad).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                s_v3 = torch.nn.functional.softmax(model_v3(x), dim=1).squeeze(0).cpu()
                s_f6 = torch.nn.functional.softmax(model_f6(x), dim=1).squeeze(0).cpu()
            if is_cont:
                return s_v3[:, :n_len], s_f6[:, :n_len]
            else:
                return s_v3[:, 4:604], s_f6[:, 4:604]

        def get_mask(sv3, sf6, length):
            pv3, pf6 = sv3.argmax(0).numpy(), sf6.argmax(0).numpy()
            mask = np.zeros(length, dtype=np.int64)
            mask[pf6 == 3] = 3
            mask[pv3 == 1] = 1
            mask[pv3 == 2] = 2
            return mask

        def render_metrics_tables(gt_mask, masks_dict):
            import pandas as pd
            if np.sum(gt_mask) == 0:
                st.info("Ground Truth vuoto. Metriche non disponibili.")
                return
            res = {}
            for name, mask in masks_dict.items():
                res[name] = compute_metrics_for_app(mask, gt_mask)

            def make_df(c_idx, c_name):
                data = []
                for name in masks_dict.keys():
                    m = res[name]
                    f1 = m[f'f1_{c_name}']
                    se = m[f'se_{c_name}']
                    ppv = m[f'ppv_{c_name}']
                    f1_mac = m['f1_macro']
                    data.append([name, f"{f1_mac:.2f}%", f"{se:.2f}%", f"{ppv:.2f}%", f"{f1:.2f}%"])
                return pd.DataFrame(data, columns=["Metodo / Fase", "F1 Macro", "Sensitivity", "Precision (PPV)", "F1-score"])

            st.markdown("##### 🔵 Metriche Onda P")
            st.table(make_df('P', 'p'))
            st.markdown("##### 🔴 Metriche Complesso QRS")
            st.table(make_df('QRS', 'q'))
            st.markdown("##### 🟢 Metriche Onda T")
            st.table(make_df('T', 't'))

        if is_ge and median_sigs and selected_lead in median_sigs:
            st.markdown("### Analisi Battito Medio (1.2s)")
            median_raw = median_sigs[selected_lead]
            
            lead_scores_v3, lead_scores_f6 = {}, {}
            for l, sig in median_sigs.items():
                s3, s6 = run_inference(sig, is_cont=False)
                lead_scores_v3[l] = s3
                lead_scores_f6[l] = s6
            
            sv3_single, sf6_single = lead_scores_v3[selected_lead], lead_scores_f6[selected_lead]
            mask_single_pre = get_mask(sv3_single, sf6_single, MEDIAN_LEN)
            
            sv3_avg = torch.stack(list(lead_scores_v3.values()), dim=0).mean(dim=0)
            sf6_avg = torch.stack(list(lead_scores_f6.values()), dim=0).mean(dim=0)
            mask_avg_pre = get_mask(sv3_avg, sf6_avg, MEDIAN_LEN)
            
            lead_ii_key = next((k for k in lead_scores_v3.keys() if k.lower() == 'ii'), None)
            if lead_ii_key:
                mask_ii_pre = get_mask(lead_scores_v3[lead_ii_key], lead_scores_f6[lead_ii_key], MEDIAN_LEN)
            else:
                mask_ii_pre = mask_avg_pre.copy()
            
            mask_single_post = postprocess_mask(mask_single_pre, fs=TARGET_FS)
            mask_avg_post = postprocess_mask(mask_avg_pre, fs=TARGET_FS)
            mask_ii_post = postprocess_mask(mask_ii_pre, fs=TARGET_FS)
            
            med_mask = mask_single_pre
            med_post = mask_single_post
            
            gt_med_mask = build_median_gt_mask(global_ann)
            fig_med = generate_plot(median_raw, gt_med_mask, med_mask, med_post, 
                                    "Battito Medio (1.2s)", selected_lead)
            st.pyplot(fig_med)

            st.markdown("#### Metriche sul Battito Medio (Tolleranza 150ms)")
            masks_dict = {
                "Singola - Pre-PP (Fase 2)": mask_single_pre,
                "Singola - Post-PP (Fase 6)": mask_single_post,
                "Avg12 - Pre-PP": mask_avg_pre,
                "Avg12 - Post-PP": mask_avg_post,
                "Lead II - Pre-PP": mask_ii_pre,
                "Lead II - Post-PP": mask_ii_post
            }
            render_metrics_tables(gt_med_mask, masks_dict)
            
            p_onset = global_ann.get('P_Onset') or global_ann.get('POnset')
            t_onset = global_ann.get('T_Onset') or global_ann.get('TOnset')
            if p_onset is None:
                st.caption("⚠️ **Nota**: L'onda P non è annotata in questo referto GE. L'F1 per l'Onda P risulterà 0% se la rete la predice.")
            if t_onset is None:
                st.caption("⚠️ **Nota**: L'inizio dell'onda T (T_Onset) non è annotato in questo referto GE (la T viene collegata al QRS). La metrica sull'onda T potrebbe risentirne.")

        st.markdown("### Tracciato Completo (10s)")
        cont_raw = cont_sigs.get(selected_lead)
        
        if cont_raw is not None:
            n_len = len(cont_raw)
            if is_ge:
                pred_mask = stamp_median_mask(beat_pos, sample_rate, med_mask, n_len)
                pred_post = stamp_median_mask(beat_pos, sample_rate, med_post, n_len)
                gt_mask = build_continuous_mask_from_ann(beat_pos, sample_rate, global_ann, n_len)
                
                max_samples = min(n_len, 10000)
                sig_plot = cont_raw[:max_samples]
                gt_mask_plot = gt_mask[:max_samples]
                pred_mask_plot = pred_mask[:max_samples]
                pred_post_plot = pred_post[:max_samples]
                
                fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                         "Tracciato Completo", selected_lead)
                st.pyplot(fig_cont)
                st.info("Le metriche AAMI continue non sono mostrate per i file GE, in quanto il Ground Truth GE e le predizioni vengono 'stampati' sugli stessi punti di ancoraggio temporali, rendendo la metrica falsata (QRS e T al 100%) e le onde P spesso assenti nel referto originale.")
                
            else:
                lead_scores_v3, lead_scores_f6 = {}, {}
                for l, sig in cont_sigs.items():
                    s3, s6 = run_inference(sig, is_cont=True)
                    lead_scores_v3[l] = s3
                    lead_scores_f6[l] = s6
                
                sv3_single, sf6_single = lead_scores_v3[selected_lead], lead_scores_f6[selected_lead]
                mask_single_pre = get_mask(sv3_single, sf6_single, n_len)
                
                sv3_avg = torch.stack(list(lead_scores_v3.values()), dim=0).mean(dim=0)
                sf6_avg = torch.stack(list(lead_scores_f6.values()), dim=0).mean(dim=0)
                mask_avg_pre = get_mask(sv3_avg, sf6_avg, n_len)
                
                lead_ii_key = next((k for k in lead_scores_v3.keys() if k.lower() == 'ii'), None)
                if lead_ii_key:
                    mask_ii_pre = get_mask(lead_scores_v3[lead_ii_key], lead_scores_f6[lead_ii_key], n_len)
                else:
                    mask_ii_pre = mask_avg_pre.copy()
                
                mask_single_post = postprocess_mask_multibeat(mask_single_pre, fs=TARGET_FS)
                mask_avg_post = postprocess_mask_multibeat(mask_avg_pre, fs=TARGET_FS)
                mask_ii_post = postprocess_mask_multibeat(mask_ii_pre, fs=TARGET_FS)
                
                pred_mask = mask_single_pre
                pred_post = mask_single_post
                gt_mask = ludb_gt_masks[selected_lead]
                
                max_samples = min(n_len, 5000)
                sig_plot = cont_raw[:max_samples]
                gt_mask_plot = gt_mask[:max_samples]
                pred_mask_plot = pred_mask[:max_samples]
                pred_post_plot = pred_post[:max_samples]
                
                fig_cont = generate_plot(sig_plot, gt_mask_plot, pred_mask_plot, pred_post_plot, 
                                         "Tracciato Completo", selected_lead)
                st.pyplot(fig_cont)
                
                st.markdown("---")
                st.markdown("### Metriche AAMI (Tolleranza 150ms)")
                
                gt_is_empty = np.sum(gt_mask_plot) == 0
                if not gt_is_empty:
                    masks_dict = {
                        "Singola - Pre-PP (Fase 2)": pred_mask_plot,
                        "Singola - Post-PP (Fase 6)": pred_post_plot,
                        "Avg12 - Pre-PP": mask_avg_pre[:max_samples],
                        "Avg12 - Post-PP": mask_avg_post[:max_samples],
                        "Lead II - Pre-PP": mask_ii_pre[:max_samples],
                        "Lead II - Post-PP": mask_ii_post[:max_samples]
                    }
                    render_metrics_tables(gt_mask_plot, masks_dict)
                else:
                    st.info("Carica le annotazioni Ground Truth per calcolare e visualizzare le metriche per questo tracciato.")

if __name__ == "__main__":
    main()
