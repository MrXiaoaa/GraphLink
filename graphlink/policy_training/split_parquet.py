#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a GraphLink policy-training parquet into train/val parquet files.")
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--val_parquet", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in (0, 1)")

    df = pd.read_parquet(args.input_parquet)
    if df.empty:
        raise RuntimeError(f"Empty parquet: {args.input_parquet}")
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * args.val_ratio)))
    val = shuffled.iloc[:n_val]
    train = shuffled.iloc[n_val:]
    if args.max_val is not None:
        val = val.iloc[: args.max_val]
    if args.max_train is not None:
        train = train.iloc[: args.max_train]
    Path(args.train_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_parquet).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.train_parquet, index=False)
    val.to_parquet(args.val_parquet, index=False)
    print({"input": len(df), "train": len(train), "val": len(val), "train_parquet": args.train_parquet, "val_parquet": args.val_parquet})


if __name__ == "__main__":
    main()
