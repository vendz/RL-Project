#!/usr/bin/env python3
"""
Process only the missing graphs from the full dataset.
Designed to run after main job times out to complete coverage.
"""

import subprocess
import sys
import os
import glob
import argparse
import csv
import re

def get_completed_graphs():
    """Get set of completed graph IDs from results shards."""
    completed = set()
    shard_files = glob.glob('results/shard_*.csv')

    for shard_file in shard_files:
        try:
            with open(shard_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add(int(row['graph_id']))
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")

    return completed

def get_rome_graph_ids(rome_dir):
    """Get all unique graph IDs from Rome dataset."""
    graph_ids = set()
    graphml_files = glob.glob(os.path.join(rome_dir, '*.graphml'))

    for f in graphml_files:
        basename = os.path.basename(f)
        m = re.match(r'grafo(\d+)\.', basename)
        if m:
            graph_ids.add(int(m.group(1)))

    return graph_ids

def find_missing_graphs(rome_dir):
    """Find graphs in Rome dataset that aren't in results."""
    rome_ids = get_rome_graph_ids(rome_dir)
    completed = get_completed_graphs()
    missing = sorted(list(rome_ids - completed))

    print(f"Rome graphs: {len(rome_ids)}")
    print(f"Completed: {len(completed)}")
    print(f"Missing: {len(missing)}")

    return missing

def process_missing_graphs(missing, rome_dir, output_file, device='cpu', verbose=False):
    """Process missing graphs using run_experiment.py."""

    if not missing:
        print("All graphs complete!")
        return

    print(f"\nProcessing {len(missing)} missing graphs...")

    # Group into ranges for efficiency
    ranges = []
    start = missing[0]
    prev = missing[0]

    for g in missing[1:]:
        if g != prev + 1:
            ranges.append((start, prev))
            start = g
        prev = g
    ranges.append((start, prev))

    print(f"Processing as {len(ranges)} ranges:\n")

    for i, (start, end) in enumerate(ranges):
        print(f"[{i+1}/{len(ranges)}] Graphs {start}-{end}...")
        cmd = [
            'python', '-m', 'scripts.run_experiment',
            '--rome-dir', rome_dir,
            '--start', str(start),
            '--end', str(end),
            '--output', output_file,
            '--device', device
        ]

        if verbose:
            cmd.append('--verbose')

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"⚠️  Range {start}-{end} failed (return code {result.returncode})")
        else:
            print(f"✓ Range {start}-{end} complete")

def main():
    parser = argparse.ArgumentParser(
        description='Process missing graphs from Rome dataset'
    )
    parser.add_argument('--rome-dir', default='rome', help='Rome dataset directory')
    parser.add_argument('--output', default='results/missing_results.csv',
                       help='Output CSV file')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Find missing
    missing = find_missing_graphs(args.rome_dir)

    if missing:
        process_missing_graphs(missing, args.rome_dir, args.output,
                              args.device, args.verbose)
        print(f"\n✅ Missing graphs processed to {args.output}")
    else:
        print("\n✅ All graphs complete!")

if __name__ == '__main__':
    main()
