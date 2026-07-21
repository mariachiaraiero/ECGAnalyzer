<div align="center">
  <h1> ECG Ensemble Viewer</h1>
  <p><strong>A Streamlit application for validating deep learning ECG segmentation on GE Mac2000 XML records.</strong></p>
</div>

## Descrizione del Progetto

Questa applicazione Streamlit fornisce un'interfaccia visiva avanzata per l'analisi e la validazione clinica delle predizioni di reti neurali (architetture U-Net) sui tracciati ECG. È progettata per interfacciarsi nativamente con il formato esportato dai dispositivi **GE Mac2000**.

Il tool permette di confrontare visivamente il **"Ground Truth"** (le misurazioni native fornite dall'algoritmo GE 12SL) con le predizioni prodotte dai modelli neurali addestrati, analizzando l'effetto del post-processing sia sul tracciato continuo (10 secondi) che sul singolo battito mediano (Median Template).

## Funzionalità Principali

- **Parsing XML Nativo:** Estrazione diretta di tracciati a 10s, battiti mediani, TOC (Time of Completion) e annotazioni cliniche dai file `.xml` GE.
- **Confronto a 3 Pannelli (Tracciato 10s):**
  1. Ground Truth (Annotazioni GE proiettate matematicamente sui battiti reali).
  2. Modello Diretto (Output crudo della rete neurale U-Net *Senza Post-Processing*).
  3. Modello Post-Processing (Predizione pulita tramite regole fisiologiche, gap-merging e filtraggio rumore).
- **Filtro Qualità (Self-Template LOO):** Esclusione automatica di battiti anomali o artefatti nel tracciato a 10s calcolando la correlazione di Pearson rispetto a un template dinamico.
- **Analisi del Singolo Battito:** Zoom sul battito mediano (1200ms) con visualizzazione dettagliata delle ampiezze (Q, R, S, J, T) ricavate tramite reverse-engineering delle logiche di baseline GE.
- **Validazione Clinica (Metriche):** Tabelle interattive che calcolano lo scarto in millisecondi tra le latenze predette dalla rete e il Ground Truth (PR, QRS, QT), con colorazione dinamica in base alle tolleranze cliniche preimpostate.

## Installazione

1. Clona questo repository:
   ```bash
   git clone https://github.com/mariachiaraiero/ECGAnalyzer.git
   cd ECGAnalyzer
   ```

2. Crea e attiva un ambiente virtuale (consigliato):
   ```bash
   python -m venv venv
   # Su Windows:
   venv\Scripts\activate
   # Su macOS/Linux:
   source venv/bin/activate
   ```

3. Installa le dipendenze:
   *(Assicurati di aver installato i pacchetti necessari per far girare i modelli e la UI)*
   ```bash
   pip install streamlit torch numpy matplotlib scipy pandas wfdb
   ```

## Utilizzo

Per avviare l'applicazione in locale, assicurati che i pesi dei modelli (es. `best_fase2.pt`, `best_fase6.pt`) siano nella directory principale, quindi esegui:

```bash
streamlit run app.py
```

Il browser si aprirà automaticamente all'indirizzo `http://localhost:8501`. 
Tramite la sidebar potrai:
- Caricare il file `.xml` del paziente.
- Selezionare le derivazioni (Leads) da ispezionare (I, II, V1-V6, ecc.).
- Scegliere l'ensemble di modelli.
- Regolare i parametri di filtraggio e attivare la visualizzazione delle misurazioni in mV.

## Architettura del Codice

- `app.py`: Applicazione principale Streamlit e UI. Gestisce il parsing dell'XML, il calcolo degli ancoraggi R-peak, il rendering dei plot `matplotlib` e la costruzione delle tabelle metriche.
- `test_ensemble_fasi_post.py`: Contiene la definizione delle architetture di rete neurale (`EnhancedUNetV3`, `ECGUNetFase6`) e ospita la funzione fondamentale `postprocess_mask` per la pulizia delle predizioni.
- `picchi_final.py`: Gestisce le logiche avanzate per l'estrazione delle ampiezze dei picchi dai segnali medi, simulando il posizionamento della baseline dell'algoritmo nativo di GE.
- `ludb_parser.py`: Modulo opzionale per il parsing di database pubblici come LUDB.


---
*Progettato per la validazione visuale rapida, il testing clinico e il debugging delle architetture di Deep Learning applicate ai segnali elettrocardiografici complessi.*
