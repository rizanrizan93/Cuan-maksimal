# IDX Emir Autonomous Scanner v1.5.1

Scanner clean-room berbasis kerangka publik Emir dengan alur **ticker-only**, database-first, persistent source cache, incremental refresh, dan exact readback.

## Hotfix v1.5.1

v1.5.0 berhasil menulis tabel besar, tetapi readback menghitung panjang body REST yang dibatasi PostgREST hingga 1.000 baris. Akibatnya tabel dengan lebih dari 1.000 baris dapat salah ditandai `ROW_COUNT_MISMATCH`. v1.5.1 memakai `Prefer: count=exact` dan `Content-Range`, sehingga verifikasi tetap exact tanpa mengunduh seluruh baris. Tidak ada migration database baru.

## Perubahan utama

v1.5.1 mempertahankan dua cache Supabase:

- `cak_ohlcv_cache`: satu payload OHLCV terkompresi secara struktur per ticker; maksimum periode kerja 3/5 tahun.
- `cak_source_cache`: cache KSEI, fundamental, dan news per ticker/family.

Alur scan:

```text
CSV ticker
→ database v6 preflight
→ baca persistent cache
→ gunakan cache yang masih valid
→ refresh tail OHLCV / sumber yang kedaluwarsa
→ tulis cache yang berubah
→ exact key + SHA-256 readback
→ hitung scanner
→ tulis 7 tabel hasil scan
→ exact scan_id readback
→ publish hasil
```

Jika cache atau hasil scan gagal diverifikasi, hasil tidak diterbitkan.

## Cache policy

| Keluarga | TTL utama | Perilaku setelah kedaluwarsa |
|---|---:|---|
| OHLCV | 12 jam | Ambil tail mulai 14 hari sebelum bar terakhir lalu merge/deduplikasi |
| KSEI profile/actions | 24 jam | Refresh; last-known-good dapat dipakai sebagai stale fallback terbatas |
| Fundamental | 7 hari | Refresh hanya ticker deep-review |
| News | 2 jam | Ambil berita terbaru, merge, deduplikasi, batasi history |

`Paksa refresh seluruh cache` tersedia untuk recovery, bukan penggunaan normal.

## Instalasi

1. Unggah seluruh isi ZIP ke root repository.
2. Di Supabase SQL Editor, jalankan migration v1 sampai v6 untuk instalasi baru. Untuk database v5 yang sudah sehat, jalankan hanya `database/migration_v6.sql`.
3. Jalankan `database/verify_v6.sql`.
4. Pastikan Streamlit Secrets:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
SUPABASE_URL = "https://PROJECT-REF.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

5. Reboot Streamlit. Target preflight: `HEALTHY_EMIR_DATABASE_V6` dan `9/9 tables readable`.
6. Jalankan cold scan 10 ticker, lalu scan kedua untuk memastikan `CACHE_HIT`.
7. Setelah itu jalankan 400 ticker.

## Status publikasi yang valid

```text
CACHE_DATABASE_COMMITTED
DATABASE_FIRST_COMMITTED
VERIFIED_ALL_TABLES
observed_status = VERIFIED_COMMITTED
```

## Batasan

- Broker inventory dan bid-offer tetap proxy OHLCV/EOD, bukan feed langsung.
- Cache mempercepat pengambilan ulang; cache tidak memperbaiki data provider yang salah.
- Fresh live-provider acceptance tetap harus dilihat dari Provider Audit di deployment Streamlit.
- Cache OHLCV 400 × 760 bar pada validasi sintetis memakai payload sekitar 28 MB sebelum overhead JSONB/Postgres.
