"""
Merge shard CSVs into a single results file.
Usage: python merge_results.py --pattern "results/shard_*.csv" --output rl_results_full.csv
"""
import argparse
import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/shard_*.csv")
    parser.add_argument("--output", default="rl_results_full.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched: {args.pattern}")
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="graph_id")
    df = df.sort_values("graph_id").reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Merged {len(files)} shards → {len(df)} graphs → {args.output}")


if __name__ == "__main__":
    main()
