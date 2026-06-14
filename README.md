# IDX Dual Tab Scanner + Research Lab

Scanner ini punya 4 lapis utama:

1. **Market Structure** — ranking cepat kandidat saham.
2. **Institutional Forward Score** — deep dive teknikal + fundamental + news catalyst.
3. **Walk-Forward Lab** — uji edge dari data OHLCV yang kamu upload.
4. **OHLCV Downloader** — download batch OHLCV IDX dari universe CSV untuk dipakai di Walk-Forward Lab.

---

## Cara menjalankan di lokal

Install dependencies:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

---

## Alur kerja yang disarankan

### 1) Download OHLCV dari universe IDX
Masuk ke tab **OHLCV Downloader**:

- upload universe CSV
- pilih periode, misalnya `1y`
- pilih interval, biasanya `1d`
- klik **Download OHLCV batch**

Hasil yang dibuat:
- CSV per ticker
- `download_results.csv`
- `download_summary.json`
- `download_manifest.json`
- ZIP bundle siap diunduh

### 2) Jalankan walk-forward
Masuk ke tab **Walk-Forward Lab**:

- upload file OHLCV hasil downloader
- pilih dataset
- atur:
  - train bars
  - test bars
  - step bars
  - min trades
- klik **Run walk-forward test**

Metrik yang dihitung:
- winrate
- expectancy (R)
- profit factor
- max drawdown
- hasil per fold

### 3) Simpan dan audit hasil
Setelah test selesai, kamu bisa:
- download `folds.csv`
- download `summary.csv`
- download `summary.json`

Di tab yang sama ada juga bagian **Import Walk-Forward Results** untuk upload balik:
- `summary.json`
- `summary.csv`
- `folds.csv`
- `trades.csv`
- ZIP bundle hasil export

Ini berguna untuk audit ulang dan perbandingan antar setting.

---

## Format universe CSV

Paling simpel cukup 1 kolom:

```csv
Ticker
AADI
ADMR
ADRO
AKRA
...
```

Kalau kolomnya bukan `Ticker`, sistem akan mencoba pakai kolom pertama.

---

## Output folder

Saat dijalankan lokal, hasil penelitian disimpan ke folder temporary aplikasi:

- `research_outputs/ohlcv_downloads/...`
- `research_outputs/...`

Saat dijalankan di Streamlit Cloud, file lokal tidak dijadikan storage permanen. Karena itu setiap bundle juga disediakan sebagai file unduhan langsung dari browser.

---

## Command line helper

### Download OHLCV batch
```bash
python ohlcv_downloader.py \
  --universe top_200_energy_basic_industrials_property.csv \
  --outdir research_outputs/ohlcv_downloads \
  --period 1y \
  --interval 1d
```

### Jalankan edge lab / walk-forward dari CLI
```bash
python run_edge_lab.py --tickers "BMRI,BBCA,TLKM,ASII" --months 24 --walkforward --out edge_lab_results.csv
```

---

## Deployment ke GitHub / Streamlit Cloud

Letakkan file-file ini di root repo:

- `app.py`
- `requirements.txt`
- `data_engine.py`
- `technical_analyst.py`
- `fundamental_analyst.py`
- `catalyst_nlp.py`
- `idx_edge_lab.py`
- `research_io.py`
- `ohlcv_downloader.py`
- `run_edge_lab.py`

Lalu deploy biasa ke Streamlit Cloud dari repo GitHub.

---

## Catatan penting

- Batch download bergantung pada Yahoo Finance.
- Untuk IDX, ticker biasanya dipakai dengan suffix `.JK`.
- Universe yang terlalu sempit bisa membuat optimasi bias sektor, jadi sebaiknya uji beberapa universe dan beberapa regime pasar.
- Walk-forward jauh lebih penting daripada hanya melihat backtest satu periode.


---

## Live Portfolio ke Supabase

Untuk storage live ledger, set environment variable berikut di Streamlit secrets atau environment deploy:

```toml
# .streamlit/secrets.toml
PORTFOLIO_DB_URL = "postgresql://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require"
```

Atau gunakan key lain yang diakui app:

- `SUPABASE_DB_URL`
- `DATABASE_URL`

Alur yang disarankan:

1. Buat project Supabase.
2. Ambil connection string Postgres dari dashboard.
3. Simpan ke `st.secrets`.
4. Deploy ulang Streamlit.
5. Buka tab **Live Portfolio** dan pastikan backend terbaca sebagai `supabase-postgres`.

Catatan:
- SQLite lokal tetap berguna untuk dev dan backup manual.
- Supabase/Postgres dipakai sebagai source of truth untuk order, fill, event, dan posisi.
