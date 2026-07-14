import re
import sys

file_path = r"c:\Users\maria\Desktop\TESI_DRIVE\APP\app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Remove slider and diagnosis from sidebar
# lines around 745:
#         tol_ms = st.sidebar.slider("Tolleranza Metriche AAMI (ms)", min_value=0, max_value=150, value=150, step=5)
#         show_peaks = st.sidebar.checkbox("Visualizza Picchi (P, Q, R, S, T, J)", value=True)
#         matched = get_matched_classes(statements) if is_ge else ludb_matched
#         if not is_ge:
#             st.sidebar.markdown("---")
#             ...

for i, line in enumerate(lines):
    if 'tol_ms = st.sidebar.slider' in line:
        lines[i] = '        tol_ms = 150\n'
    if 'if not is_ge:' in line and 'st.sidebar.markdown("---")' in lines[i+1]:
        # remove this entire block up to st.title
        j = i
        while 'st.title("Visualizzatore Ensemble ECG")' not in lines[j]:
            lines[j] = ''
            j += 1

# 2. At line 824: st.markdown("---")
# We will insert the columns and side diagnosis
col_setup = """        st.markdown("---")
        
        col_main, col_side = st.columns([4, 1])
        with col_side:
            if not is_ge:
                st.markdown("### Diagnosi Paziente")
                st.markdown(f"**Macro-Classe:** `{ludb_main_class}`")
                if ludb_matched:
                    st.markdown(f"**Classi attivate:** {', '.join(ludb_matched)}")
                
                with st.expander("Diagnosi Originali", expanded=True):
                    if ludb_diag_lines:
                        for d in ludb_diag_lines:
                            st.write(f"- {d}")
                    else:
                        st.write("Nessuna diagnosi trovata")
            else:
                st.markdown("### Diagnosi Paziente")
                st.markdown(f"**Classi GE attivate:** {', '.join(matched) if matched else 'Standard'}")
                with st.expander("Statement Originali", expanded=True):
                    if statements:
                        for s in statements:
                            st.write(f"- {s}")
                    else:
                        st.write("Nessuna diagnosi trovata")
"""

start_plot_idx = -1
for i, line in enumerate(lines):
    if 'st.markdown("---")' in line and 'if not is_ge:' in lines[i+2]:
        lines[i] = col_setup
        start_plot_idx = i + 1
        break

# 3. Replace st. with col_main. in the plotting section
for i in range(start_plot_idx, len(lines)):
    lines[i] = lines[i].replace('st.markdown(', 'col_main.markdown(')
    lines[i] = lines[i].replace('st.pyplot(', 'col_main.pyplot(')
    lines[i] = lines[i].replace('st.success(', 'col_main.success(')
    lines[i] = lines[i].replace('container=st', 'container=col_main')

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
