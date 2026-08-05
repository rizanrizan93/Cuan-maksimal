from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from autonomous_enrichment import ksei_actions_to_events, ksei_profiles_to_maps
from data_providers import assess_benchmark_freshness
from narrative_flow_engine import score_narrative_events

BASE = Path('/mnt/data')


def main() -> None:
    provider = pd.read_csv(BASE / '2026-08-04T04-47_export-1.csv')
    integrity_old = pd.read_csv(BASE / '2026-08-04T04-46_export-1.csv')
    thesis_old = pd.read_csv(BASE / '2026-08-04T04-46_export-4.csv')

    ksei_errors = provider[(provider['provider'] == 'KSEI_SECURITY_PROFILE') & (provider['status'] == 'ERROR')]
    failed_profiles = pd.DataFrame({
        'ticker': ksei_errors['ticker'].tolist(),
        'ksei_source_url': ['https://web.ksei.co.id/provider-error'] * len(ksei_errors),
        'ksei_source_verified': [False] * len(ksei_errors),
        'security_status': [np.nan] * len(ksei_errors),
    })
    _, corrected_integrity = ksei_profiles_to_maps(failed_profiles, pd.DataFrame(), as_of='2026-08-04')
    corrected_rows = list(corrected_integrity.values())

    admin_actions = pd.DataFrame([
        {'ticker': 'TEST.JK', 'action_type': 'Cash Dividend', 'record_date': '2026-07-01', 'distribution_date': '2026-07-10', 'status': 'Active', 'source_url': 'https://web.ksei.co.id/test'},
        {'ticker': 'TEST.JK', 'action_type': 'Proxy Voting', 'record_date': '2026-07-02', 'distribution_date': '2026-07-11', 'status': 'Active', 'source_url': 'https://web.ksei.co.id/test'},
    ])
    admin_events = ksei_actions_to_events(admin_actions, as_of='2026-08-04')
    admin_score = score_narrative_events(admin_events, as_of='2026-08-04')

    ohlcv = provider[(provider['audit_family'] == 'OHLCV') & (provider['status'] == 'OK')]
    frames = {}
    for _, row in ohlcv.iterrows():
        date = pd.to_datetime(row['last_date'], errors='coerce')
        if pd.notna(date):
            frames[str(row['ticker'])] = pd.DataFrame({'Close': [1.0]}, index=pd.DatetimeIndex([date]))
    benchmark_row = provider[(provider['ticker'] == '^JKSE') & (provider['audit_family'] == 'BENCHMARK')].iloc[0]
    benchmark_date = pd.to_datetime(benchmark_row['last_date'])
    benchmark = pd.DataFrame({'Close': [1.0]}, index=pd.DatetimeIndex([benchmark_date]))
    benchmark_gate = assess_benchmark_freshness(benchmark, frames, min_universe_count=20)

    old_false_suspensions = integrity_old[
        integrity_old['ticker'].isin(ksei_errors['ticker']) & integrity_old['suspension_flag'].fillna(False)
    ]
    old_admin_titles = thesis_old['narrative_latest_title'].fillna('').str.startswith('KSEI corporate action:').sum()
    old_uniform_fundamental = int((pd.to_numeric(thesis_old['fundamental_coverage_pct'], errors='coerce') == 83.3).sum())

    result = {
        'source_scan_rows': int(len(integrity_old)),
        'ksei_provider_errors_in_source_scan': int(len(ksei_errors)),
        'old_false_suspension_rows_linked_to_ksei_error': int(len(old_false_suspensions)),
        'v150_provider_error_suspension_unknown': int(sum(pd.isna(row['suspension_flag']) for row in corrected_rows)),
        'v150_provider_error_hard_blocks': int(sum(bool(row['idx_integrity_hard_block']) for row in corrected_rows)),
        'old_ksei_admin_titles_used_as_narrative': int(old_admin_titles),
        'v150_admin_event_narrative_count': int(admin_score['narrative_event_count']),
        'v150_admin_event_state': admin_score['narrative_state'],
        'old_uniform_fundamental_coverage_83_3_rows': old_uniform_fundamental,
        'benchmark_gate': benchmark_gate,
        'assertions': {
            'ksei_error_never_becomes_suspension': all(pd.isna(row['suspension_flag']) for row in corrected_rows),
            'ksei_error_never_becomes_hard_block': all(not bool(row['idx_integrity_hard_block']) for row in corrected_rows),
            'administrative_events_excluded_from_narrative': admin_score['narrative_event_count'] == 0,
            'stale_benchmark_rejected': benchmark_gate['benchmark_usable'] is False,
        },
    }
    if not all(result['assertions'].values()):
        raise SystemExit(json.dumps(result, indent=2))
    output = Path('validation_artifacts/LIVE_SCAN_REGRESSION_AUDIT_V1_5_0.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
