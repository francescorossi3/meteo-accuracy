@echo off
REM Prova completa con DATI SINTETICI (Windows, ambiente conda attivo)
cd /d "%~dp0"
set METEOITA_DATA=demo_data
python scripts\demo_generate.py || exit /b 1
for /f %%d in ('python -c "import datetime as d;print((d.date.today()-d.timedelta(80)).isoformat())"') do set FROM=%%d
python scripts\03_verify.py --from %FROM% || exit /b 1
python scripts\04_train_gpr.py || exit /b 1
python scripts\05_build_maps.py --demo || exit /b 1
python scripts\06_build_site.py || exit /b 1
echo Apri http://localhost:8000  (Ctrl+C per chiudere)
python -m http.server -d build\site 8000
