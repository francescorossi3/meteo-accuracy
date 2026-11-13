# Meteo Accuracy

Confronto dell'accuratezza dei provider di previsioni meteo **in Italia e in tutto il mondo** (soprattutto **pioggia** e **temperatura**),
per previsioni **orarie, a 6 ore e giornaliere**, con anticipo da **0 a 7 giorni**, più tre stime combinate:

| Stima | Come funziona |
|---|---|
| ★ **Migliore** | per ogni caso (variabile × aggregazione × anticipo) usa il provider con l'errore più basso negli ultimi 90 giorni |
| ⚖ **Media pesata** | media dei provider con pesi ∝ 1/MSE |
| ◆ **GPR** | regressione a processo gaussiano: da posizione, quota e previsioni di tutti i modelli produce una stima **e la sua incertezza**, in qualsiasi punto |

**Regioni** (ciascuna con classifiche, pesi, modelli GPR e mappe propri; si aggiungono/attivano in `config.yaml`):

| Regione | Modelli | Stazioni | Mappe | Default |
|---|---|---|---|---|
| **Italia** | tutti quelli che coprono l'Italia: ICON-2I (2 km), ICON-EU, ARPEGE, KNMI + i globali | ~60 sinottiche | griglia 0.35° | attiva |
| **RadCalNet** | globali: ECMWF IFS e AIFS, ICON, GFS, UKMO, GEM | nessuna: **solo archivio** delle previsioni sui 5 siti (Railroad Valley, Gobabeb, La Crau, Baotou ×2) | i 5 siti (media e dispersione dei modelli) | attiva |
| **Mondo** | globali | ~200 su tutti i continenti | griglia 6° | **spenta** (troppe chiamate per il piano gratuito Open-Meteo) |

Per RadCalNet il sito mostra **solo i dati dei modelli** (ogni modello, media e dispersione), senza verifica né GPR;
le previsioni orarie di ogni modello sui siti, per ogni anticipo 0–7 giorni, sono **salvate ogni giorno** in
`data/csv/radcalnet/AAAA-MM.csv` (e nei parquet di `data/forecasts`), pronte per il confronto con le misure dei siti.

Cliccando un punto del sito la regione si sceglie da sola: Italia → modelli ad alta risoluzione e combinazioni verificate;
entro 150 km da un sito RadCalNet o altrove → solo i modelli globali e la loro media.
Giorni e blocchi di 6 ore sono sempre nell'**ora locale** del punto o della stazione.

Il sito (GitHub Pages) mostra classifiche, la verifica sugli eventi significativi (pioggia forte, caldo, gelo), mappe di stima e incertezza su tutta Italia, la verifica di affidabilità del GPR,
e permette di **cliccare un punto ovunque nel mondo o inserire lat/lon** per ottenere la previsione combinata in quel punto.
Tutti i dati sono nel repository in **CSV** (+ un riepilogo **Excel**).

```
 Open-Meteo (ICON-2I, ECMWF, ICON, GFS, ARPEGE, UKMO, ...)  ─┐                ┌─► classifiche CSV/Excel/JSON
 MET Norway, OpenWeatherMap, WeatherAPI, Tomorrow, VC  ─────┼─► coppie prev./oss. ─┼─► migliore + pesi
 Meteostat (stazioni SYNOP italiane) ───────────────────────┘   1h · 6h · 24h      ├─► GPR (validato su holdout)
                                                                 lead 0…7 giorni    └─► mappe + sito
```

Tutto gira gratis su **GitHub Actions** una volta al giorno: nessun server da mantenere.

---

## 1. Messa in funzione (circa 10 minuti)

1. **Crea il repository** su GitHub (es. `meteo-accuracy`, pubblico: Actions e Pages sono gratuiti senza limiti di minuti).
2. **Carica i file** di questa cartella:
   ```bash
   cd meteo-accuracy
   git init && git add . && git commit -m "Primo commit"
   git branch -M main
   git remote add origin https://github.com/francescorossi3/meteo-accuracy.git
   git push -u origin main
   ```
3. **Permessi**: *Settings → Actions → General → Workflow permissions* → **Read and write permissions**.
4. **Pages**: *Settings → Pages → Build and deployment → Source* → **GitHub Actions**.
5. (Facoltativo) **Chiavi API** dei servizi commerciali in *Settings → Secrets and variables → Actions*:
   `OWM_API_KEY`, `WEATHERAPI_KEY`, `TOMORROW_API_KEY`, `VISUALCROSSING_KEY` (tutti hanno un piano gratuito).
   Senza chiave il provider viene semplicemente saltato.
6. **Storico**: *Actions → Ricostruzione storico → Run workflow* con un intervallo di **~2 mesi** (es. 2026-09-01 → 2026-10-31).
   Ripeterlo nei giorni successivi per andare più indietro (dati dei modelli disponibili dal 2024).
   I servizi «snapshot» non hanno storico: si accumula dal primo giorno di funzionamento.
7. **Primo aggiornamento**: *Actions → Aggiornamento giornaliero → Run workflow*. Poi parte da solo ogni mattina.
   Il sito sarà su `https://francescorossi3.github.io/meteo-accuracy/`.

> **Repository di lavoro / organizzazione**: se il repository è di un'organizzazione, un amministratore deve permettere
> GitHub Actions e Pages (*Organization settings → Actions* e *→ Pages*). Su repository **privati** di un piano gratuito
> Pages non è disponibile e i minuti di Actions sono limitati (2.000/mese, l'aggiornamento giornaliero ne usa ~20–40 al giorno):
> conviene un repository pubblico o un piano GitHub Team/Enterprise.

### Ogni quanto devo intervenire?
Mai, di norma: il workflow **Aggiornamento giornaliero** parte da solo ogni mattina sui server di GitHub (anche a computer spento),
aggiorna dati, classifiche, modelli e mappe e ripubblica il sito. Se un giorno fallisce GitHub manda una mail e il sito resta
all'ultimo aggiornamento riuscito.

> Al primo avvio guarda i log del workflow: i nomi dei modelli Open-Meteo e i formati dei dati Meteostat
> cambiano ogni tanto. Se un modello dà errore viene saltato e lo vedi nel log (`! icon_eu: ...`): basta
> correggere il campo `model` in `config.yaml` o metterlo `enabled: false`.

## 2. Uso in locale (conda / Spyder)

```bash
conda create -n meteoita python=3.12 -y && conda activate meteoita
pip install -r requirements.txt
```
Gli script in `scripts/` sono lineari (niente `main()`), quindi in Spyder le variabili restano nel Variable Explorer.
I parametri si passano da riga di comando o si cambiano i default in testa allo script.

| Script | Cosa fa |
|---|---|
| `00_setup_stations.py` | per ogni regione sceglie N stazioni ben distribuite → `data/stations.csv` (modificabile a mano; contiene anche il fuso orario) |
| `01_fetch_forecasts.py` | previsioni sulle stazioni (`--start --end --mode openmeteo` per lo storico) |
| `02_fetch_observations.py` | osservazioni orarie |
| `03_verify.py` | coppie, statistiche, classifiche, migliore e pesi, CSV |
| `04_train_gpr.py` | modelli GPR + validazione |
| `05_build_maps.py` | mappe su griglia (`--demo` = dati sintetici) |
| `06_build_site.py` | assembla il sito in `build/site` e il file Excel |

**Prova senza internet** (dati inventati, solo per vedere il sito): `run_demo.bat` su Windows (o `bash run_demo.sh`) e poi apri http://localhost:8000. Richiede ~15 minuti (addestramento GPR).

## 3. I dati (cartella `data/`)

| File | Contenuto |
|---|---|
| `csv/riepilogo.xlsx` | tutte le tabelle principali in un file Excel |
| `csv/classifica.csv` | metriche per provider × variabile × aggregazione × anticipo × finestra (30/90/365 giorni, tutto) |
| `csv/classifica_holdout.csv` | ultimi 14 giorni, con Migliore, Media pesata e GPR (scelti/addestrati *prima* di quei giorni) |
| `csv/migliore_e_pesi.csv` | provider migliore e pesi in uso |
| `csv/gpr_validazione.csv` | MAE, RMSE, bias, CRPS, copertura 68% e 95% del GPR |
| `csv/trend_mensile.csv` | metriche mese per mese |
| `csv/eventi_punteggi.csv` | verifica per eventi significativi: osservati, colti, mancati, falsi allarmi, POD/FAR/CSI, errore nei giorni di evento |
| `csv/radcalnet/AAAA-MM.csv` | previsioni orarie di ogni modello sui siti RadCalNet (colonne `t2m_<modello>`, `prcp_<modello>`), per sito e anticipo |
| `csv/eventi_elenco.csv` | ogni evento osservato (giorno × stazione) con la previsione di ciascun provider a ogni anticipo |
| `csv/coppie_giornaliere/AAAA-MM.csv` | per giorno/stazione/variabile/anticipo: osservato e previsione di ogni provider |
| `stats/daily_stats.parquet` | somme giornaliere da cui si ricava qualsiasi metrica su qualsiasi periodo |
| `forecasts/<provider>/AAAA-MM.parquet` | previsioni orarie archiviate (compatte) |
| `observations/AAAA-MM.parquet` | osservazioni orarie |

CSV con separatore virgola e punto decimale (in Excel italiano: *Dati → Da testo/CSV*). I parquet si leggono con
`pd.read_parquet` o con le funzioni `meteoita.storage.read_forecasts / read_observations` (che riportano i valori in °C e mm).

**Metriche**: bias, MAE, RMSE, correlazione; per la pioggia anche POD, FAR, CSI, ETS e frequency bias a tre soglie
(1h: 0.2/1/5 mm · 6h: 0.5/2/10 mm · 24h: 1/5/20 mm, modificabili in `config.yaml`).

## 4. Metodo

**Regioni.** Una stazione appartiene a tutte le regioni che la contengono (le italiane valgono anche per il Mondo).
Un modello entra in una regione solo se il suo dominio (`coverage` in `config.yaml`) la copre tutta: per il Mondo
restano quindi i modelli globali. Il GPR usa la posizione come punto sulla sfera (x, y, z), così funziona anche a
cavallo dell'antimeridiano e ai poli.

**Anticipo (lead day).** Per i modelli Open-Meteo si usa la *Previous Runs API*: `lead 0` = run più recente, `lead N` = valore
previsto N×24 h prima dell'ora valida. Per i servizi «snapshot» (archiviati alle ~06 UTC) `lead N` = giorno di calendario
locale (0 oggi, 1 domani, …): l'anticipo medio è qualche ora diverso, da tenere presente nel confronto.

**Aggregazioni.** Ora di fine periodo come in tutte le fonti; blocchi di 6 ore (00–06, 06–12, …) e giorni nell'ora locale
della stazione. Nelle mappe del Mondo i giorni sono locali di ogni punto, le mappe orarie e a 6 ore sono in UTC.
Temperatura: media (e per il giorno anche massima e minima); pioggia: somma. Un blocco si verifica solo se completo
(previsione e osservazione sulle stesse ore). Oltre il 2° giorno MET Norway e OpenWeatherMap hanno passo 3–6 h:
per loro non si verifica l'aggregazione oraria a quegli anticipi.

**Eventi significativi.** Gli errori medi sono dominati dai giorni «normali»; per questo si verificano a parte gli eventi
osservati alle stazioni (default: pioggia ≥ 20 e ≥ 50 mm/giorno, Tmax ≥ 35 °C, Tmin ≤ 0 °C; si aggiungono in `config.yaml`
alla voce `events`). Per ogni provider e anticipo: quanti eventi ha colto (POD), quanti falsi allarmi (FAR), il CSI
e l'errore sul valore nei giorni di evento. La scheda «Eventi» del sito mostra anche l'elenco degli ultimi eventi con le
previsioni di ciascuno (verde = colto, rosso = mancato).

**Migliore e media pesata** sono scelti solo tra i modelli disponibili ovunque (Open-Meteo), così si possono calcolare su qualsiasi
punto e sulle mappe; i servizi «solo stazioni» compaiono comunque nelle classifiche.

**GPR.** Un modello per variabile × aggregazione × anticipo. Feature: lat, lon, quota, previsioni dei modelli
(pioggia in radice quadrata). Kernel `C·RBF(ARD) + DotProduct + White`: la parte lineare impara una combinazione dei modelli,
la RBF le correzioni spaziali e non lineari, il rumore bianco l'errore residuo, che entra nell'incertezza.
Si addestra sugli ultimi 45 giorni con al massimo 300 punti (configurabile) e si esporta in JSON:
il sito calcola stima e incertezza **nel browser** con le stesse formule (`web/gp.js` ≡ `meteoita/gp.py`).

**Le stime GPR sono giuste?** Ogni giorno il GPR viene riaddestrato *escludendo* gli ultimi 14 giorni e verificato su questi:
- errore (MAE/RMSE) confrontato con il miglior provider e con la media pesata sugli stessi giorni;
- **calibrazione**: l'osservato deve cadere nell'intervallo 68% circa il 68% delle volte e nel 95% circa il 95%.

Il risultato è nella scheda «Affidabilità GPR» del sito, in `gpr_validazione.csv` e come cerchi sulle mappe (errore per stazione).

## 5. Scalabilità

- **Aggiungere un provider**: una riga in `config.yaml` (modello Open-Meteo) oppure, per un'altra API, una funzione
  `fetch_<api>(lat, lon, elev, key)` in `meteoita/providers.py` registrata in `SNAPSHOT_APIS`.
- **Più stazioni / reti regionali (ARPA)**: aggiungere righe a `data/stations.csv` e una sorgente in `meteoita/observations.py`.
  Le reti ARPA hanno molte stazioni anche in montagna e pluviometri migliori: è il miglioramento più utile.
- **Nuova regione** (es. Europa, Alpi, un paese): una voce in `regions` con poligono GeoJSON, numero di stazioni e passo di griglia.
- **Spazio**: le previsioni sono salvate come interi a 16 bit in parquet compressi (~250 MB/anno con ~260 stazioni),
  le classifiche come somme giornaliere: per anni di dati il repository resta piccolo. Oltre ~1 GB conviene spostare
  `data/forecasts` su *GitHub Releases* o Hugging Face Datasets (cambia solo `meteoita/storage.py`).
- **Limiti delle API gratuite**: Open-Meteo ~10.000 chiamate/giorno ed è gratuito solo per uso **non commerciale**
  (attribuzione CC BY 4.0). Con le impostazioni di default (Italia + archivio RadCalNet) l'aggiornamento giornaliero ne usa circa 4.000;
  attivando il Mondo si sale a ~8.000, vicino al limite (stima: Open-Meteo
  conta più chiamate per richieste con molti punti o variabili). Se nei log compare l'errore 429 ("limit exceeded"):
  ridurre `n_stations` o allargare `grid_res`, oppure — e per un uso **di lavoro/commerciale** è comunque necessario — prendere
  un piano Open-Meteo a pagamento e mettere la chiave nel Secret `OPENMETEO_API_KEY` (il codice usa allora gli endpoint `customer-`).
  Tomorrow.io gratuito è limitato a 20 stazioni.
- **Mappe più fini**: `grid_res` di ogni regione (più punti = più chiamate API).

## 6. Limiti da conoscere

- **Siti italiani come 3BMeteo, ilMeteo, Meteo.it** non hanno API pubbliche gratuite e i loro termini vietano lo scraping:
  non sono inclusi. Se ottieni un accesso API (3BMeteo ha un'offerta B2B) basta aggiungere una funzione `fetch_...`.
- Le osservazioni orarie di pioggia delle stazioni sinottiche non sono complete ovunque: il numero di coppie `n` è sempre riportato.
- Le stazioni sono quasi tutte aeroporti in pianura/costa: in montagna classifiche e incertezze GPR sono meno rappresentative.
- I siti RadCalNet non sono verificati automaticamente (le stazioni sinottiche più vicine sono troppo lontane o in climi diversi).
  Le previsioni archiviate si possono confrontare con le misure meteo dei siti; se le aggiungete come sorgente di
  osservazioni basta mettere `verify: true` per avere classifiche, pesi e GPR anche lì.
- Il Mondo, se attivato, ha stazioni rade (Africa, oceani): il GPR va letto con la sua incertezza.
- I valori dei modelli sono medie su celle di griglia, le stazioni misurano un punto: parte dell'errore è di rappresentatività.

## Licenze e attribuzioni
Codice: MIT. Dati dei modelli: [Open-Meteo](https://open-meteo.com) (CC BY 4.0) e i rispettivi servizi meteorologici
(ItaliaMeteo-ARPAE, ECMWF, DWD, NOAA, Météo-France, UK Met Office, ECCC, KNMI). Osservazioni: [Meteostat](https://meteostat.net).
Librerie del sito: Leaflet (BSD-2), Chart.js (MIT), mappe © OpenStreetMap.
