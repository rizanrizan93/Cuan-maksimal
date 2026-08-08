from __future__ import annotations
import json
import pandas as pd
from narrative_flow_engine import build_emir_profile, ENGINE_VERSION


def inputs(i:int, direct:bool=False):
    features={"feature_state":"OK","smart_money_score":80,"smart_money_coverage_pct":100,"market_structure_score":80,"market_structure_mode":"CONTINUATION_SETUP","trend_score":82,"liquidity_score":75,"distribution_score":10,"crowding_score":30,"execution_friction_score":10,"price_stage":"BASE_TRANSITION","absorption_score":80,"last_price":1000+i,"ema20":970+i,"high20":1020+i,"low20":930+i,"atr14":30,"adtv20_idr":1_000_000_000,"gap_risk_score":0,"ohlcv_integrity_state":"VALID","corporate_action_anomaly_flag":False}
    narrative={"narrative_score":78,"narrative_coverage_pct":90,"narrative_state":"MATERIAL_THESIS_CONFIRMED","narrative_verified_source_count":1,"narrative_independent_story_count":2,"financial_conversion_score":75,"issuer_alignment_score":80,"issuer_alignment_coverage_pct":90,"story_runway_score":80,"top_down_catalyst_score":75,"industry_translation_score":75,"retail_adoption_stage":"PRE_RETAIL"}
    broker={"broker_inventory_score":75,"broker_inventory_coverage_pct":80,"broker_inventory_shift_state":"COLLECTION","retail_cannibalisation_risk":0}
    orderbook={"orderbook_trigger_score":75,"orderbook_coverage_pct":90 if direct else 65,"precise_trigger_price":1020+i,"orderbook_provenance_state":"DIRECT_SOURCE_VERIFIED" if direct else "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH"}
    integrity={"idx_integrity_score":90,"idx_integrity_coverage_pct":90,"idx_integrity_hard_block":False,"idx_integrity_unknown_critical_count":0,"idx_integrity_provenance_state":"DIRECT_SOURCE_VERIFIED" if direct else "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS","corporate_action_review_cleared":True}
    fundamental={"fundamental_conversion_score":78,"fundamental_coverage_pct":90,"fundamental_data_quality_score":90,"fundamental_official_source_coverage_pct":85,"fundamental_cashflow_state":"IDX_OFFICIAL_YTD_OCF_FCF_AVAILABLE","fundamental_period_freshness_state":"CURRENT_QUARTERLY_PERIOD","fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","fundamental_state":"FUTURE_FUNDAMENTAL_SUPPORTIVE"}
    market={"market_regime":"SELECTIVE","market_context_score":65,"market_context_coverage_pct":100}
    sector={"sector_leadership_score":70,"sector_context_coverage_pct":100,"sector_rrg_state":"LEADING","sector_relative_strength_pct":5,"sector_strength_momentum_pct":2}
    return features,narrative,broker,orderbook,integrity,fundamental,market,sector


def main():
    rows=[]
    for i in range(400):
        direct = i == 0
        f,n,b,o,integ,fun,m,s=inputs(i,direct)
        rows.append(build_emir_profile(ticker=f"T{i:03d}.JK",features=f,narrative=n,broker=b,ownership={"ownership_score":70,"ownership_coverage_pct":70},orderbook=o,market=m,sector=s,integrity=integ,fundamental=fun,deep_reviewed=True,capital_mode="GUARDED_REAL_MONEY",risk_budget_pct=2.0,max_position_cap_pct=20.0))
    frame=pd.DataFrame(rows)
    proxy=frame[frame["orderbook_provenance_state"].eq("OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH")]
    direct=frame[frame["orderbook_provenance_state"].eq("DIRECT_SOURCE_VERIFIED")]
    assert int(proxy["production_ready"].fillna(False).sum()) == 0
    assert float(frame["risk_budget_pct"].max()) <= 0.75
    assert float(frame["position_cap_pct"].max()) <= 10.0
    assert int(direct["production_ready"].fillna(False).sum()) == 1
    result={"scanner_version":ENGINE_VERSION,"rows":len(frame),"eod_proxy_rows":len(proxy),"eod_proxy_production_ready":int(proxy["production_ready"].sum()),"direct_verified_rows":len(direct),"direct_verified_production_ready":int(direct["production_ready"].sum()),"max_risk_budget_pct":float(frame["risk_budget_pct"].max()),"max_position_cap_pct":float(frame["position_cap_pct"].max()),"state":"PASS"}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
