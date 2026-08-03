# IDX Narrative Flow Scanner v1.0.0

Standalone Streamlit scanner untuk saham IDX yang memusatkan analisis pada **narrative lifecycle, money flow, smart-money behavior proxy, market structure, ownership context, scenario execution, dan risk discipline**.

## Status dan batas penggunaan

Proyek ini merupakan **clean-room public-framework reconstruction**. Ia bukan produk, afiliasi, endorsement, formula proprietary, atau replika materi berbayar milik Emir Parengkuan/CAK Investment Club. Dasar desain hanya memakai prinsip yang telah disampaikan secara publik:

- narrative play dan fase akumulasi → hype → distribusi;
- money flow, volume, dan reaksi harga;
- integrasi fundamental, teknikal, money flow, serta ownership context;
- keputusan berbasis alasan/skenario dan manajemen risiko.

Rujukan publik:

- https://freeclass.cakinvestmentclub.com/
- https://cakinvestmentclub.com/
- https://mediaindonesia.com/ekonomi/859618/edukasi-trading-dibutuhkan-investor-untuk-bekal-hadapi-dinamika-pasar
- https://20.detik.com/detik-sore/20260121-260121109/video-strategi-hadapi-dampak-rumor-early-narrative-bagi-investor-pasar-saham

Klaim return/success rate pada situs atau materi promosi tidak dimasukkan ke formula scanner dan tidak dianggap sebagai track record terverifikasi.

## Arsitektur dua tahap

### 1. FAST_DISCOVERY

Memproses OHLCV seluruh universe untuk:

- trend dan relative strength;
- value traded dan liquidity;
- CMF, OBV slope, close acceptance;
- accumulation/distribution days;
- absorption/failed absorption;
- pullback volume contraction;
- crowding dan extension risk.

Mode ini cocok untuk discovery universe hingga sekitar 400 ticker. Karena tidak mengambil berita online, hasil narrative akan `EVIDENCE_PENDING` kecuali pengguna mengunggah Narrative Events CSV.

### 2. HYBRID_400_TO_DEEP

Memproses OHLCV seluruh universe, kemudian hanya kandidat teratas yang memperoleh deep public-news review. Ini mode default dan paling efisien untuk universe besar.

### 3. DEEP_REVIEW

Melakukan narrative review terhadap seluruh ticker yang diunggah. Disarankan untuk universe maksimal 100 ticker.

## Output utama

- `narrative_flow_lifecycle`
  - `FLOW_AHEAD_OF_STORY`
  - `STORY_AHEAD_OF_FLOW`
  - `ACCUMULATION_BUILDING`
  - `EARLY_NARRATIVE_FLOW_CONVERGENCE`
  - `EXPANSION_CONFIRMED`
  - `CROWDED_HYPE`
  - `DISTRIBUTION_OR_BROKEN`
- `smart_money_score`
- `narrative_score`
- `narrative_flow_conviction_score`
- `narrative_flow_coverage_pct`
- `public_method_state`
  - `PUBLIC_FRAMEWORK_READY`
  - `PUBLIC_FRAMEWORK_WATCH`
  - `PUBLIC_FRAMEWORK_EVIDENCE_PENDING`
  - `PUBLIC_FRAMEWORK_REJECT`
- `action`
- entry zone, trigger, invalidation/SL, TP1, TP2, RR;
- position cap dan lot simulation berdasarkan modal/risk budget.

## Fail-closed evidence policy

- Tidak ada news/event: narrative tetap kosong, bukan otomatis 50.
- Tidak ada ownership file: ownership tidak memengaruhi score.
- Broker code tidak dianggap beneficial owner.
- OHLCV hanya menghasilkan **smart-money behavior proxy**, bukan identifikasi bandar.
- Execution plan hanya menjadi `READY_WITH_TRIGGER` bila narrative–flow convergence, coverage, liquidity, crowding, dan distribution gate lulus.

## Format universe CSV

```csv
ticker
ADMR
MDKA
BRMS
```

Nama kolom yang diterima: `ticker`, `symbol`, `kode`, atau `code`. Ticker otomatis dinormalisasi menjadi `.JK`.

## Broker Summary CSV opsional

```csv
ticker,date,broker_code,buy_value,sell_value,source_verified,provenance_state
ADMR,2026-07-31,XX,1500000000,800000000,true,DIRECT_SOURCE_VERIFIED
```

Minimal gunakan `buy_value/sell_value` atau `buy_volume/sell_volume`. `source_verified=true` hanya boleh dipakai bila datanya benar-benar berasal dari sumber langsung yang dapat diaudit.

## Narrative Events CSV opsional

```csv
ticker,event_date,title,category,materiality_score,financial_bridge_score,source_url,source_tier
ADMR,2026-07-31,Project expansion,PROJECT_EXPANSION,80,75,https://issuer.example,OFFICIAL
```

## Ownership CSV opsional

```csv
ticker,free_float_pct,owner_alignment_score,insider_buy_flag,controller_change_flag,ownership_note
ADMR,15,70,false,false,Verified from public disclosure
```

Ownership score tidak dibentuk otomatis dari tebakan.

## Deployment Streamlit Cloud

1. Upload seluruh file root paket ke repository baru.
2. Main file path: `app.py`.
3. Deploy.
4. Untuk database opsional, jalankan `database/migration_v1.sql` di Supabase.
5. Isi Streamlit Secrets:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
SUPABASE_URL = "https://PROJECT.supabase.co"
SUPABASE_SECRET_KEY = "YOUR_SECRET_KEY"
```

Jangan menyimpan secret key di GitHub.

## Supabase persistence

Tabel terpisah dari IDX Super Scanner:

- `cak_scan_runs`
- `cak_radar_snapshots`
- `cak_narrative_events`

Tab Database memiliki write report dan exact `scan_id` readback verification.

## Performa

Komputasi teknikal bersifat vectorized per ticker dan dapat menangani 400 ticker. Waktu live tetap bergantung pada koneksi, Yahoo, rate limit, retries, serta jumlah deep-review ticker. Gunakan:

- 400 ticker: `HYBRID_400_TO_DEEP`, deep limit 20–40;
- 100 ticker: `DEEP_REVIEW` atau `HYBRID`;
- scan harian: universe prioritas;
- discovery mingguan: universe besar.

## Pengujian

```bash
pytest -q
```

Build v1.0.0 diuji menggunakan synthetic deterministic OHLCV. Network smoke test pada lingkungan build tidak dapat dijalankan karena DNS eksternal dibatasi; provider tetap memakai Yahoo direct dengan fallback yfinance saat deployment memiliki internet.
