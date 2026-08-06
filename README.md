# IDX Emir Autonomous Scanner v1.5.2

Scanner clean-room berbasis kerangka publik Emir dengan input ticker-only, persistent cache, incremental refresh, dan **best-effort database persistence**.

## Perubahan v1.5.2

Database tidak lagi menjadi hard publication gate. Scanner sekarang memakai alur:

```text
CSV ticker
→ periksa tabel cache yang dapat dibaca
→ gunakan cache valid per ticker/family
→ ambil ulang hanya data cache yang hilang, rusak, atau kedaluwarsa
→ hitung radar dan deep review
→ coba simpan perubahan cache dan hasil scan
→ tampilkan hasil walaupun persistence parsial atau database tidak tersedia
```

Kegagalan database tidak diubah menjadi data bullish atau nilai default. Data pasar/fundamental/narrative yang tidak tersedia tetap mengikuti fail-closed evidence gate pada masing-masing ticker.

## Status persistence

```text
SCAN_COMPLETED_FULL_PERSISTENCE
SCAN_COMPLETED_PARTIAL_PERSISTENCE
SCAN_COMPLETED_MEMORY_ONLY
```

- `FULL`: seluruh hasil ditulis dan exact-count verified.
- `PARTIAL`: sebagian hasil/cache tersimpan; hasil tetap ditampilkan, dan bagian yang tidak tersimpan akan diambil/dihitung ulang pada scan berikutnya.
- `MEMORY_ONLY`: database tidak tersedia; hasil tetap dapat digunakan pada session Streamlit dan diekspor, tetapi tidak persisten.

## Cache behavior

| Keluarga | Perilaku |
|---|---|
| OHLCV | cache hit per ticker; cache miss mengambil full period; cache stale mengambil tail 14 hari lalu merge |
| KSEI | cache valid 24 jam; ticker yang hilang dicari ulang |
| Fundamental | cache valid 7 hari; hanya shortlist yang direfresh |
| News | cache valid 2 jam; berita baru di-merge dan dideduplikasi |

Satu ticker cache hit tidak memaksa ticker lain memakai cache. Setiap ticker/family diputuskan secara independen.

## Database

Schema tetap `emir_autonomous_schema_v6`. Tidak ada migration baru dari v1.5.0/v1.5.1.

Secrets yang didukung:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
SUPABASE_URL = "https://PROJECT-REF.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

Database disarankan, tetapi scan tidak lagi dinonaktifkan bila database belum sehat.

## Deployment

1. Ganti isi repository dengan seluruh isi ZIP v1.5.2.
2. Commit ke branch `main`.
3. Reboot Streamlit.
4. Tidak perlu menjalankan migration baru bila schema v6 sudah ada.
5. Jalankan 10 ticker, lalu periksa tab Database dan Provider Audit.
6. Lanjutkan 300–400 ticker setelah hasil kecil stabil.

## Batasan

- Broker inventory dan bid-offer otomatis tetap proxy OHLCV/EOD.
- Hasil `MEMORY_ONLY` hilang saat session Streamlit di-reset jika tidak diunduh.
- Cache/database tidak memperbaiki data provider yang salah.
- Readback parsial bukan bukti bahwa baris yang tidak terbaca hilang; scanner tetap melaporkan jumlah write dan readback secara terpisah.
