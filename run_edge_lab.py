from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_engine import parse_universe_text
from idx_edge_lab import StrategyParams, universe_backtest, walk_forward_test, summarize_walk_forward
from data_engine import load_ticker_data, normalize_ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="IDX edge lab for scanner validation")
    parser.add_argument("--tickers", type=str, default="BMRI,BBCA,TLKM,ASII", help="Comma or newline separated tickers")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--benchmark", type=str, default="^JKSE")
    parser.add_argument("--out", type=str, default="edge_lab_results.csv")
    parser.add_argument("--walkforward", action="store_true")
    args = parser.parse_args()

    tickers = parse_universe_text(args.tickers)
    params = StrategyParams()

    if args.walkforward:
        rows = []
        benchmark = load_ticker_data(args.benchmark, args.months)
        for t in tickers:
            sym = normalize_ticker(t)
            if not sym:
                continue
            df = load_ticker_data(sym, args.months)
            folds = walk_forward_test(df, benchmark, benchmark_symbol=args.benchmark)
            summary = summarize_walk_forward(folds)
            summary["symbol"] = sym
            rows.append(summary)
        out = pd.DataFrame(rows)
    else:
        out = universe_backtest(tickers, months=args.months, benchmark_symbol=args.benchmark, params=params)

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
