import re

MACRO_CLASS_NAMES = {
    0: 'HEALTHY',
    1: 'PACING',
    2: 'NO_P',
    3: 'WIDE_QRS',
    4: 'OLD_MI',
    5: 'AV_BLOCK',
    6: 'ST_T',
    7: 'HYPERTROPHY',
    8: 'UNKNOWN',
}
# Priorità di assegnazione (prima che matcha vince)
PRIORITY = ['PACING', 'NO_P', 'WIDE_QRS', 'OLD_MI', 'AV_BLOCK', 'ST_T', 'HYPERTROPHY', 'HEALTHY']

# Keyword per macro-classe  — IN INGLESE (diagnosi del LUDB)
CLASS_MAPPING_EN = {
    'PACING': [
        'pacing', 'pacemaker',
    ],
    'NO_P': [
        'atrial fibrillation', 'atrial flutter',
        'wandering atrial pacemaker',
    ],
    'WIDE_QRS': [
        'bundle branch block', 'hemiblock',
        'intravintricular conduction delay',
        'intraventricular conduction delay',
        'aberrant conduction',
    ],
    'OLD_MI': [
        'scar formation', 'ischemia/scar/supp.nstemi',
        'stemi',
    ],
    'AV_BLOCK': [
        'av block', 'av-block',
        'sinoatrial blockade',
    ],
    'ST_T': [
        'repolarization abnormalities',
        'early repolarization syndrome',
        'ischemia:',   # "Ischemia: anterior wall" ecc.
        'overload',    # "Left ventricular overload" -> alterazioni ST-T
    ],
    'HYPERTROPHY': [
        'hypertrophy',
        'left axis deviation', 'right axis deviation',
    ],
    'HEALTHY': [
        'sinus rhythm', 'sinus bradycardia', 'sinus tachycardia',
        'sinus arrhythmia', 'irregular sinus rhythm',
    ],
}

def parse_hea_diagnoses(hea_content: str):
    """
    Estrae le linee diagnostiche direttamente dal contenuto stringa del file .hea
    """
    diag_lines = []
    lines = hea_content.splitlines()
    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            continue
        content = line.lstrip('#').strip()
        
        # Saltiamo l'età e il sesso, o tag conosciuti che non sono patologie
        if content.startswith('<age>:') or content.startswith('<sex>: '):
            continue
            
        if '<diagnoses>' not in content and content:
            diag_lines.append(content)
            
    return diag_lines

def get_macro_class_data(diag_lines):
    """
    Riceve la lista di righe diagnostiche e restituisce un set con TUTTE le macro-classi individuate
    (incluso come prioritario il main_class), usando la logica originale del dataset.
    """
    matched = set()
    txt = ' '.join(diag_lines).lower() if diag_lines else ''

    # --- Macro-classi ---
    for mc, keywords in CLASS_MAPPING_EN.items():
        if any(kw in txt for kw in keywords):
            matched.add(mc)

    # --- Classe principale per priorità ---
    main_class = 'UNKNOWN'
    for p in PRIORITY:
        if p in matched:
            main_class = p
            break
            
    return main_class, matched, diag_lines
