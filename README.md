# Aurum — Gold & Silver Forecast Dashboard

Static dashboard (`index.html`) + LSTM training script (`gold_silver_forecast.py`)
that forecasts gold (GC=F) and silver (SI=F) prices 7 days out on real Yahoo
Finance data. A GitHub Action retrains daily and commits `forecast_output.json`,
which the dashboard reads automatically.

## 1. Push to GitHub

```bash
cd aurum-dashboard
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/aurum-dashboard.git
git push -u origin main
```

## 2. Get real data into the repo

The GitHub Action runs on a daily schedule, but trigger it once manually so
`forecast_output.json` exists before you deploy:
1. Go to your repo → **Actions** tab
2. Select **Retrain forecast model** → **Run workflow**
3. Wait for it to finish (a few minutes) — it commits `forecast_output.json` back to `main`

## 3. Deploy to Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import the GitHub repo
2. Framework preset: **Other** (it's a static file, no build step needed)
3. Leave build command empty, output directory as root
4. Deploy

Every time the daily Action commits a new `forecast_output.json`, Vercel
auto-redeploys with the fresh numbers — no manual steps after setup.

## Files

- `index.html` — the dashboard UI (responsive, no build step, no dependencies)
- `gold_silver_forecast.py` — fetches real data, trains an LSTM per metal, writes `forecast_output.json`
- `requirements.txt` — Python deps for the training script
- `.github/workflows/retrain.yml` — daily retraining automation
