# Emir Scanner v1.9.27 — KSEI checkpoint guard

## Root cause fixed

A resumable KSEI chunk could be marked `CHUNK_CACHE_COMMIT_FAILED` even when
all tickers were processed. The database KSEI integrity trigger intentionally
preserves the existing cache row when a refresh contains an unresolved or
placeholder security profile. The scanner previously expected the rejected
refresh SHA, read back the preserved OLD SHA, interpreted that expected guard
outcome as a cache failure, and set the scan job to `PAUSED`.

## Runtime correction

- Mirrors the database KSEI profile validity contract before checkpoint commit.
- If an invalid refresh would be rejected and a hash-valid durable cache row
  exists, the scanner reuses that durable row and suppresses only the rejected
  write.
- Emits `STALE_CACHE_FALLBACK_GUARD` in provider audit for traceability.
- Exact SHA verification remains strict for every write that should actually
  reach storage. Real cache write/readback failures still fail safe.
- Existing resumable jobs remain compatible and can resume from the same
  stored stage/offset.

This fix does not relax KSEI evidence integrity or overwrite protected evidence.
