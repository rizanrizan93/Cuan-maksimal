# Deployment Safe Update — v1.7.0

Gunakan penggantian repository secara atomik. **Jangan hapus `app.py`** atau file root produksi lebih dahulu lalu melakukan upload pada commit terpisah, karena Streamlit dapat menarik repository pada keadaan setengah terpasang.

## Prosedur aman

1. Extract ZIP v1.7.0 di lokal.
2. Salin seluruh isi root release ke working tree repository yang sama.
3. Pastikan `app.py`, `narrative_flow_engine.py`, `autonomous_enrichment.py`, `resumable_scan.py`, `top3_dashboard.py`, dan `VERSION` sudah hadir bersamaan.
4. Commit seluruh penggantian tersebut dalam **satu commit**.
5. Push ke branch deployment (`main`).
6. Reboot Streamlit dan lakukan live smoke scan kecil sebelum 400-ticker production scan.

Database schema v7 dan secrets v1.6.4 tetap kompatibel. Jangan mencampur runtime v1.6.4 dengan analytical modules v1.7.0 karena version/cache bump sengaja digunakan untuk mencegah reuse output dengan semantik lama.
