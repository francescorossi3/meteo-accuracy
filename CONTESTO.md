# Meteo Accuracy – contesto per riprendere il lavoro (anche in una nuova chat)

> Da incollare all'inizio di una nuova conversazione con Claude, oppure chiedergli:
> «aggiungi il repository `francescorossi3/meteo-accuracy` e leggi `CONTESTO.md`».
> Ultimo aggiornamento di questo file: 14/11/2026.

## Dove sta tutto
- **Repository**: https://github.com/francescorossi3/meteo-accuracy (pubblico; l'app GitHub di Claude è installata e può fare push).
- **Sito**: https://francescorossi3.github.io/meteo-accuracy/ (GitHub Pages, interfaccia in **inglese**).
- **Copia locale**: `C:\Users\franc\Desktop\meteo-accuracy` (può essere indietro: fa fede GitHub; aggiornarla con `git pull`).
- Proprietario: Francesco (CNR-IMIOT). Script Python in stile lineare, adatti a Spyder (niente `main()`).

## Cosa fa
- Confronta l'accuratezza dei provider di previsioni per **temperatura** (media, max, min) e **pioggia**,
  aggregazione **oraria / 6 ore / giornaliera** (ora locale), anticipo **0–7 giorni**.
- **Provider**:
  - modelli Open-Meteo: ItaliaMeteo ICON-2I (3 giorni), ECMWF IFS 0.25°, ECMWF AIFS, DWD ICON-EU (5 giorni),
    ICON Global, NOAA GFS, ARPEGE Europe (4 giorni), UKMO 10 km, GEM;
  - KNMI Harmonie è disattivato perché la Previous Runs API risponde 400;
  - MET Norway, archiviato ogni giorno ("snapshot");
  - OpenWeatherMap, WeatherAPI, Tomorrow.io e Visual Crossing sono predisposti ma attivi solo con le chiavi
    API nei Secrets (oggi non ci sono).
- **Osservazioni**: circa 60 stazioni sinottiche italiane da Meteostat, scelte automaticamente e ben distribuite
  (alcune in montagna).
- **Metriche**:
  - bias, MAE, RMSE, correlazione, **R²**;
  - per la pioggia anche POD, FAR, CSI, ETS e frequency bias a tre soglie;
  - **eventi significativi**: pioggia ≥20 e ≥50 mm/giorno, Tmax ≥35 °C, Tmin ≤0 °C.
- **Stime combinate**:
  - ★ **Best**: il provider con l'errore più basso negli ultimi 90 giorni per variabile, aggregazione e anticipo;
    se in un punto manca, si usa il successivo;
  - ⚖ **Weighted mean**: pesi proporzionali a 1/MSE;
  - ◆ **GPR**: un modello per variabile × aggregazione × anticipo. Kernel `C·RBF(ARD) + DotProduct + White`;
    feature: posizione sulla sfera (x, y, z), quota e previsioni dei modelli; pioggia in radice quadrata.
    Esportato in JSON e calcolato anche nel browser (`web/gp.js`, identico a `meteoita/gp.py`, testato).
- **Validazione del GPR** (scheda "GPR reliability"):
  1. ogni giorno, sugli ultimi 14 giorni esclusi dall'addestramento: RMSE, MAE, **R²**, bias, CRPS e copertura
     degli intervalli 68% e 95%;
  2. ogni settimana, **backtest storico** (`07_backtest_gpr.py`): blocchi di 14 giorni, addestramento sui 45
     precedenti, Best e pesi scelti anch'essi ex-ante. Confronto con Best ex-ante, Weighted e "best ex-post";
     RMSE, **R²**, percentuale di blocchi vinti dal GPR, copertura.
     Serve uno storico ≥ 59 giorni: il primo risultato vero arriva pochi giorni dopo il 14/11/2026.
- **Regioni** (`config.yaml → regions`):
  - `italia`: verificata, mappe su griglia di 0,35° (276 punti);
  - `radcalnet`: Railroad Valley RVUS, Gobabeb GONA, La Crau LCFR, Baotou BSCN/BTCN, con `verify: false`.
    Sul sito solo i dati dei modelli (media e dispersione). Previsioni orarie di ogni modello archiviate in
    `data/csv/radcalnet/AAAA-MM.csv` per il confronto futuro con le misure dei siti;
  - `mondo`: pronta ma `enabled: false` (troppe chiamate per il piano gratuito Open-Meteo, ~10.000 al giorno).
- **Classifiche**: oltre ai provider includono ★ Best, ⚖ Weighted e ◆ GPR, sempre **fuori campione**
  (holdout giornaliero + blocchi del backtest salvati nello storico `daily_stats`; il GPR è su un campione di righe).
- **Sito**: Rankings, Events, Maps (best / weighted / GPR / incertezza / probabilità di pioggia / singoli modelli),
  Point forecast (ricerca **indirizzo** con OpenStreetMap Nominatim, click, lat/lon, siti RadCalNet),
  GPR reliability, Data & method.

## Struttura del codice
```
config.yaml                 tutto configurabile (provider, regioni, eventi, soglie, GPR, backfill)
meteoita/                   config, storage (parquet Int16 x10), providers, observations, verify (metriche, R²),
                            gp, regions, grid, events, demo (dati sintetici)
scripts/00_setup_stations   01_fetch_forecasts   02_fetch_observations   03_verify   04_train_gpr
        05_build_maps       06_build_site        07_backtest_gpr         08_backfill_window
        09_reuse_published_maps   demo_generate
web/                        index.html, app.js, gp.js, style.css, vendor (Leaflet, Chart.js)
data/                       forecasts/, observations/, stats/daily_stats.parquet, csv/ (CSV + riepilogo.xlsx)
tests/test_gp_consistency   sklearn = export JSON = gp.js
```
I nomi dei CSV sono in italiano: `classifica.csv`, `classifica_holdout.csv`, `migliore_e_pesi.csv`,
`gpr_validazione.csv`, `gpr_backtest(_sintesi).csv`, `trend_mensile.csv`, `eventi_*.csv`,
`coppie_giornaliere/`, `radcalnet/`, `stazioni.csv`.

## Automazione (GitHub Actions, nessun intervento necessario)
- `daily.yml`, 06:17 UTC, con **avvio di riserva alle 10:37 UTC** che parte solo se quello del mattino non ha
  aggiornato i dati. Fa: dati → verifica → GPR → backtest (se l'ultimo ha più di 7 giorni) → mappe → sito → commit → Pages.
- `backfill_auto.yml`, 13:43 UTC: aggiunge 20 giorni di storico più vecchio al giorno, fino a `backfill.start`
  (2025-06-01); stato in `data/backfill_state.json`.
- `backfill.yml`: backfill manuale su un intervallo a scelta.
- `site_only.yml` "Ricostruisci sito (senza API)": rigenera classifiche, GPR, backtest e pagine dai dati del
  repository, riusando le mappe già pubblicate (`09_reuse_published_maps.py`): **zero chiamate alle API meteo**.
- I workflow condividono `concurrency: data-writes`, quindi non scrivono mai i dati in contemporanea.

## Storia e note
- 13/11/2026: creato il progetto, primo run reale riuscito. Lo storico parte dal 21/10/2026 e cresce ogni giorno.
- 14/11/2026: il cron del mattino è partito in ritardo e l'ho lanciato a mano. Aggiunti l'avvio di riserva e l'R².
- Vincoli: Open-Meteo gratuito solo per uso non commerciale. Con un piano a pagamento la chiave va nel Secret
  `OPENMETEO_API_KEY` e il codice usa da solo gli endpoint `customer-`.
- Limiti noti: stazioni quasi tutte in pianura o costa; RadCalNet non verificato (mancano le misure dei siti).

## Possibili sviluppi
- Aggiungere le misure meteo dei siti RadCalNet come osservazioni e mettere `verify: true`.
- Aggiungere le stazioni delle reti ARPA regionali (più stazioni e montagna).
- Attivare la regione `mondo` o una regione `europa` (meglio con la chiave Open-Meteo).
- Aggiungere le chiavi API dei servizi commerciali per includerli nelle classifiche.
