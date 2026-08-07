from __future__ import annotations
import copy, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import resumable_scan as rs
from persistence import DatabaseConfig

TICKERS = [
    'ELSA.JK','RAJA.JK','ADMR.JK','MDKA.JK','SSIA.JK','TAPG.JK','IMPC.JK','RMKE.JK','BRPT.JK','DSNG.JK',
    'EXCL.JK','MBSS.JK','NICL.JK','WIFI.JK','BSDE.JK','SGER.JK','PKPK.JK','SMIL.JK','FINN.JK','MFIN.JK'
]

def frame(seed:int,bars:int=320,end='2026-08-05'):
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range(end=end,periods=bars)
    rets=rng.normal(0.0005,0.018,bars)
    close=100*np.exp(np.cumsum(rets))
    op=close*(1+rng.normal(0,0.004,bars))
    hi=np.maximum(op,close)*(1+rng.uniform(0.001,0.015,bars))
    lo=np.minimum(op,close)*(1-rng.uniform(0.001,0.015,bars))
    vol=rng.integers(500_000,5_000_000,bars)
    return pd.DataFrame({'Open':op,'High':hi,'Low':lo,'Close':close,'Volume':vol},index=dates)

def main():
    universe=[{'ticker':t,'company_name':t.replace('.JK',''),'sector':'TEST'} for t in TICKERS]
    frames={t:frame(i+1) for i,t in enumerate(TICKERS[:-4])}
    frames['PKPK.JK']=frame(17,bars=80)
    frames['SMIL.JK']=frame(18,bars=125)
    # FINN, MFIN intentionally unavailable
    frames['^JKSE']=frame(999,bars=320,end='2026-08-03')  # stale versus universe
    profiles={}; actions=defaultdict(list); news=defaultdict(list); fundamentals={}; chunks=[]; calls=defaultdict(list)
    job={
      'scan_id':'integration20','universe_hash':'h20','scanner_version':'1.6.3','status':'CREATED','current_stage':'BENCHMARK',
      'current_offset':0,'current_chunk':0,'chunk_size':5,'total_tickers':20,'processed_tickers':0,'failed_tickers':0,
      'progress_pct':0.0,'scan_mode':'EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP','result_status':'PENDING','universe':universe,
      'settings':{'scan_mode':'EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP','period':'5y','completed_only':True,'workers':3,
        'deep_limit':10,'news_per_ticker':4,'use_google_news':True,'use_yahoo_news':True,'auto_ksei':True,
        'auto_fundamental':True,'force_cache_refresh':False,'capital':5_000_000,'risk_pct':1.0,'max_position_cap_pct':20.0,
        'calibration_mode':'SHADOW_ONLY'},'shortlist':[],'failures':{},'result_summary':{},'last_error':''}
    cache_fail_once={'done':False}

    def update(_c,_sid,patch):
      nonlocal job
      job={**job,**copy.deepcopy(dict(patch))}; return copy.deepcopy(job)
    def record(_c,**kwargs):
      x=copy.deepcopy(kwargs); x['payload']=copy.deepcopy(kwargs.get('payload') or {}); chunks.append(x); return x
    def fetch_ohlcv(_c,requested,**kwargs):
      requested=list(requested); calls['OHLCV'].append(requested)
      out={t:frames[t] for t in requested if t in frames}
      audit=pd.DataFrame([{'ticker':t,'provider':'FIXTURE','status':'CACHE_HIT' if t in out else 'PROVIDER_FAILED'} for t in requested])
      writes=[{'ticker':t,'content_sha256':'x'} for t in out]
      return out,audit,writes
    def persist_cache(_c,**kwargs):
      count=len(kwargs.get('ohlcv_rows') or [])+len(kwargs.get('source_rows') or [])
      # one transient failure on first universe OHLCV chunk; should retry same chunk
      if count and not cache_fail_once['done'] and any(r.get('ticker')!='^JKSE' for r in (kwargs.get('ohlcv_rows') or [])):
        cache_fail_once['done']=True
        return (pd.DataFrame([{'table':'__SUMMARY__','rows_attempted':count,'rows_written':0,'state':'CACHE_WRITE_PARTIAL'}]),
                pd.DataFrame([{'table':'__SUMMARY__','rows_expected':count,'rows_verified':0,'state':'CACHE_DATABASE_NOT_COMMITTED'}]))
      return (pd.DataFrame([{'table':'__SUMMARY__','rows_attempted':count,'rows_written':count,'state':'CACHE_WRITE_ALL'}]),
              pd.DataFrame([{'table':'__SUMMARY__','rows_expected':count,'rows_verified':count,'state':'CACHE_DATABASE_COMMITTED'}]))
    def load_ohlcv(_c,requested,**kwargs):
      requested=list(requested); out={t:frames[t] for t in requested if t in frames}
      return out,pd.DataFrame([{'ticker':t,'provider':'FIXTURE_CACHE','status':'CACHE_LOAD' if t in out else 'CACHE_MISS'} for t in requested])
    def fetch_ksei(_c,requested,**kwargs):
      req=list(requested); calls['KSEI'].append(req); p=[]; a=[]; audit=[]; writes=[]
      for t in req:
        if t in {'PKPK.JK','SMIL.JK'}:
          audit.append({'ticker':t,'provider':'KSEI_FIXTURE','status':'ERROR','detail':'HTTP 500 fixture'})
          continue
        item={'ticker':t,'company_name':t.replace('.JK',' Tbk'),'sector':'TEST','provider_state':'OK'}
        profiles[t]=item; p.append(item); audit.append({'ticker':t,'provider':'KSEI_FIXTURE','status':'OK'}); writes.append({'cache_key':f'KSEI:{t}','content_sha256':'x'})
      return pd.DataFrame(p),pd.DataFrame(a),pd.DataFrame(audit),writes
    def load_ksei(_c,requested):
      p=[profiles[t] for t in requested if t in profiles]
      return pd.DataFrame(p),pd.DataFrame(),pd.DataFrame()
    def fetch_news(_c,local,**kwargs):
      req=local['ticker'].tolist(); calls['NEWS'].append(req); ev=[]; audit=[]; writes=[]
      for t in req:
        if t in {'ELSA.JK','RAJA.JK','ADMR.JK'}:
          audit.append({'ticker':t,'provider':'NEWS_FIXTURE','status':'NO_ITEMS'}); writes.append({'cache_key':f'NEWS:{t}','content_sha256':'empty'}); continue
        item={'ticker':t,'published_at':pd.Timestamp('2026-08-05',tz='UTC'),'title':f'Expansion {t}','summary':'capacity expansion',
          'publisher':'Fixture','url':f'https://example.com/{t}','source_tier':'MEDIA','source_verified':True,'materiality_score':70,
          'financial_bridge_score':60,'top_down_catalyst_score':60,'industry_translation_score':60,'issuer_alignment_score':60,
          'category':'PROJECT_CAPACITY'}
        news[t].append(item); ev.append(item); audit.append({'ticker':t,'provider':'NEWS_FIXTURE','status':'OK'}); writes.append({'cache_key':f'NEWS:{t}','content_sha256':'x'})
      return pd.DataFrame(ev),pd.DataFrame(audit),writes
    def load_news(_c,requested):
      return pd.DataFrame([x for t in requested for x in news.get(t,[])]),pd.DataFrame()
    def fetch_fund(_c,requested,**kwargs):
      req=list(requested); calls['FUNDAMENTAL'].append(req); rows=[]; audit=[]; writes=[]
      for t in req:
        if t in {'PKPK.JK','SMIL.JK','WIFI.JK'}:
          audit.append({'ticker':t,'provider':'FUND_FIXTURE','status':'NO_DATA'}); continue
        x={'ticker':t,'fundamental_state':'FUTURE_FUNDAMENTAL_SUPPORTIVE','fundamental_coverage_pct':80,
           'fundamental_conversion_score':72,'revenue_growth_pct':20,'earnings_growth_pct':25}
        fundamentals[t]=x; rows.append(x); audit.append({'ticker':t,'provider':'FUND_FIXTURE','status':'OK'}); writes.append({'cache_key':f'FUNDAMENTAL:{t}','content_sha256':'x'})
      return pd.DataFrame(rows),pd.DataFrame(audit),writes
    def load_fund(_c,requested):
      return pd.DataFrame([fundamentals[t] for t in requested if t in fundamentals]),pd.DataFrame()
    def load_chunks(_c,_sid): return pd.DataFrame(chunks)
    def persist_result(_c,**kwargs):
      radar=kwargs['radar']; n=len(radar)
      write=pd.DataFrame([
        {'table':'__SUMMARY__','rows_attempted':n+10,'rows_written':n+8,'state':'WRITE_PARTIAL'},
        {'table':'cak_radar_snapshots','rows_attempted':n,'rows_written':n,'state':'WRITTEN'},
        {'table':'cak_autonomous_evidence','rows_attempted':10,'rows_written':8,'state':'WRITE_PARTIAL'}])
      verify=pd.DataFrame([{'table':'__SUMMARY__','state':'PARTIAL_READBACK','verification_pct':90.0}])
      commit=pd.DataFrame([{'state':'SCAN_COMPLETED_PARTIAL_PERSISTENCE','publishable':True}])
      return write,verify,commit

    rs.update_scan_job=update; rs.record_job_chunk=record; rs.fetch_ohlcv_cache_first=fetch_ohlcv
    rs.persist_verify_cache_bundle=persist_cache; rs.cache_commit_succeeded=lambda v: (not v.empty and str(v.iloc[0].get('state'))=='CACHE_DATABASE_COMMITTED')
    rs.load_cached_ohlcv_frames=load_ohlcv; rs.fetch_ksei_cache_first=fetch_ksei; rs.load_cached_ksei=load_ksei
    rs.fetch_news_cache_first=fetch_news; rs.load_cached_news=load_news; rs.fetch_fundamental_cache_first=fetch_fund; rs.load_cached_fundamentals=load_fund
    rs.load_job_chunks=load_chunks; rs.persist_verify_scan_best_effort=persist_result

    cfg=DatabaseConfig(True,'https://fixture.supabase.co','sb_secret_fixture',key_type='SECRET')
    reports=[]; final=None
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      for step in range(40):
        job,report,result=rs.process_next_job_step(cfg,copy.deepcopy(job),now=pd.Timestamp('2026-08-06 15:00',tz='Asia/Jakarta'))
        reports.append({'step':step+1,'stage_after':job.get('current_stage'),'status':job.get('status'),'report':report.get('state')})
        if step==2:
          job=json.loads(json.dumps(job,default=str))
        if result is not None: final=result
        if job.get('status') in {'COMPLETED','COMPLETED_PARTIAL_PERSISTENCE'}: break
      warning_text=[str(w.message) for w in caught]

    assert final is not None, job
    radar=final['radar']
    assert len(radar)==20, len(radar)
    assert set(radar['ticker'])==set(TICKERS)
    failed_states=radar.set_index('ticker').loc[['FINN.JK','MFIN.JK'],'emir_decision_state'].tolist()
    assert failed_states == ['EMIR_DATA_INTEGRITY_BLOCK', 'EMIR_DATA_INTEGRITY_BLOCK'], failed_states
    assert job['status']=='COMPLETED_PARTIAL_PERSISTENCE',job
    assert job['progress_pct']==100.0
    assert cache_fail_once['done']
    # Same first OHLCV chunk should have been attempted twice due to transient cache failure.
    universe_calls=[x for x in calls['OHLCV'] if x!=['^JKSE']]
    assert len(universe_calls)>=5 and universe_calls[0]==universe_calls[1],universe_calls
    mean_warn=[x for x in warning_text if 'Mean of empty slice' in x]
    date_warn=[x for x in warning_text if 'Parsing dates' in x]
    assert final['benchmark_freshness'].get('benchmark_freshness_state') == 'STALE_RELATIVE_TO_UNIVERSE'
    assert not mean_warn
    artifact_dir = Path(__file__).resolve().parent / 'validation_artifacts_v1_6_3'
    artifact_dir.mkdir(exist_ok=True)
    export = radar.copy()
    export.insert(0, 'validation_data_mode', 'CONTROLLED_FIXTURE_NOT_LIVE_MARKET_DATA')
    export.to_csv(artifact_dir / 'TWENTY_TICKER_RADAR_SOURCE_V1_6_3.csv', index=False)
    summary = {
      'state':'PASS','tickers_input':20,'radar_rows':len(radar),'steps':len(reports),'job_status':job['status'],
      'shortlist_count':len(job.get('shortlist') or []),'ohlcv_failed':job.get('failures',{}).get('OHLCV',[]),
      'ksei_calls':sum(map(len,calls['KSEI'])),'news_calls':sum(map(len,calls['NEWS'])),'fundamental_calls':sum(map(len,calls['FUNDAMENTAL'])),
      'transient_cache_retry':universe_calls[0]==universe_calls[1],'mean_empty_warnings':len(mean_warn),'date_parse_warnings':len(date_warn),
      'benchmark_state':final['benchmark_freshness'].get('benchmark_freshness_state'),
      'decision_counts':radar['emir_decision_state'].value_counts().to_dict(),
      'reports':reports
    }
    (artifact_dir / 'TWENTY_TICKER_VALIDATION_V1_6_3.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
