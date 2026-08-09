import pandas as pd
import research_memory as rm
from persistence import DatabaseConfig

class _Resp:
    def __init__(self, payload): self._payload=payload
    def json(self): return self._payload

def _rows(n=5):
    return [{"memory_id":f"m{i}","content_sha256":f"h{i}","ticker":"T.JK","family":"X","payload":{}} for i in range(n)]

def test_research_memory_retries_transient_chunk_and_verifies(monkeypatch):
    rows=_rows(5); calls={}
    def post(config, *, table, conflict, payload, chunk_size, return_rows):
        key=payload[0]["memory_id"]; calls[key]=calls.get(key,0)+1
        if key=="m2" and calls[key]==1: raise RuntimeError("503 transient")
        return len(payload)
    def request(config, method, table, **kwargs):
        # return all expected hashes; enough to test readback state
        return _Resp([{"memory_id":r["memory_id"],"content_sha256":r["content_sha256"]} for r in rows])
    monkeypatch.setattr(rm,"_post_payload_in_chunks",post)
    monkeypatch.setattr(rm,"_request",request)
    monkeypatch.setattr(rm.time,"sleep",lambda *_: None)
    cfg=DatabaseConfig(True,"https://fixture.supabase.co","k")
    write, verify=rm.persist_verify_research_memory(cfg,scan_id="s",rows=rows,chunk_size=2)
    assert write.iloc[0]["state"]=="RESEARCH_MEMORY_WRITTEN"
    assert int(write.iloc[0]["rows_written"])==5
    assert verify.iloc[0]["state"]=="RESEARCH_MEMORY_VERIFIED_EXACT"
    assert int(verify.iloc[0]["rows_verified"])==5

def test_research_memory_does_not_lie_when_all_writes_fail(monkeypatch):
    rows=_rows(3)
    monkeypatch.setattr(rm,"_post_payload_in_chunks",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("403 schema/RLS")))
    monkeypatch.setattr(rm,"_request",lambda *a,**k: _Resp([]))
    monkeypatch.setattr(rm.time,"sleep",lambda *_: None)
    cfg=DatabaseConfig(True,"https://fixture.supabase.co","k")
    write, verify=rm.persist_verify_research_memory(cfg,scan_id="s",rows=rows,chunk_size=2)
    assert write.iloc[0]["state"]=="RESEARCH_MEMORY_WRITE_FAILED"
    assert int(write.iloc[0]["rows_written"])==0
    assert verify.iloc[0]["state"]!="RESEARCH_MEMORY_VERIFIED_EXACT"
    assert int(verify.iloc[0]["rows_verified"])==0
