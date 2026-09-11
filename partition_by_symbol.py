#!/usr/bin/env python3
"""
Streaming Parquet Repartitioner
Converts monolithic 1.6 GB shards (train-*.parquet) into dedicated per-symbol Parquet files
under data/by_symbol/{symbol}.parquet for sub-10ms instantaneous market data queries.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("partitioner")

INPUT_DIR = config.DATA_DIR / "1min"
OUTPUT_DIR = config.DATA_DIR / "by_symbol"


def flush_symbol(sym: str, chunks: List[pa.Table], out_dir: Path):
    """Flushes buffered PyArrow table chunks for a given symbol to its parquet file."""
    if not sym or not chunks:
        return
    
    # Strip any illegal characters for file safety
    safe_sym = sym.replace("/", "_").replace("\\", "_").strip()
    target_file = out_dir / f"{safe_sym}.parquet"
    
    new_tbl = pa.concat_tables(chunks) if len(chunks) > 1 else chunks[0]
    
    if target_file.exists():
        try:
            old_tbl = pq.read_table(target_file)
            combined = pa.concat_tables([old_tbl, new_tbl])
            # Sort by timestamp column
            sorted_indices = pc.sort_indices(combined["timestamp"])
            combined = combined.take(sorted_indices)
            pq.write_table(combined, target_file, compression="snappy")
            return
        except Exception as e:
            logger.warning(f"Error merging with existing file for {safe_sym}: {e}")

    pq.write_table(new_tbl, target_file, compression="snappy")


def run_partitioning():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(INPUT_DIR.glob("train-*.parquet"))
    
    if not shards:
        logger.error(f"No train-*.parquet shards found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(shards)} Parquet shards in {INPUT_DIR}")
    
    # Calculate total row groups
    total_rgs = 0
    total_est_rows = 0
    shard_info = []
    for s in shards:
        pf = pq.ParquetFile(s)
        rg_count = pf.num_row_groups
        rows = pf.metadata.num_rows
        shard_info.append((s, rg_count, rows))
        total_rgs += rg_count
        total_est_rows += rows

    logger.info(f"Total Rows: {total_est_rows:,} across {total_rgs} row groups")
    logger.info(f"Output Directory: {OUTPUT_DIR}")

    t_start = time.time()
    current_sym: Optional[str] = None
    current_chunks: List[pa.Table] = []
    processed_rgs = 0
    processed_rows = 0
    symbols_written = set()

    for s_idx, (shard_path, num_rgs, s_rows) in enumerate(shard_info, 1):
        pf = pq.ParquetFile(shard_path)
        logger.info(f"[{s_idx}/{len(shard_info)}] Opening {shard_path.name} ({s_rows:,} rows, {num_rgs} row groups)...")
        
        for rg_idx in range(num_rgs):
            rg = pf.read_row_group(rg_idx)
            rg_len = len(rg)
            processed_rows += rg_len
            processed_rgs += 1
            
            # Find symbol boundaries using contiguous numpy change points (ultra-fast)
            sym_col = rg["symbol"].to_numpy(zero_copy_only=False)
            change_idx = np.where(sym_col[:-1] != sym_col[1:])[0] + 1
            slices = [0] + list(change_idx) + [rg_len]
            
            for i in range(len(slices) - 1):
                st, en = slices[i], slices[i + 1]
                sym = sym_col[st]
                
                if sym != current_sym:
                    if current_sym is not None and current_chunks:
                        flush_symbol(current_sym, current_chunks, OUTPUT_DIR)
                        symbols_written.add(current_sym)
                    current_sym = sym
                    current_chunks = []
                
                current_chunks.append(rg.slice(st, en - st))

            # Telemetry every 25 row groups
            if processed_rgs % 25 == 0 or processed_rgs == total_rgs:
                elapsed = time.time() - t_start
                rate = processed_rows / max(0.1, elapsed)
                rem_rows = max(0, total_est_rows - processed_rows)
                eta_s = rem_rows / max(1.0, rate)
                pct = (processed_rows / total_est_rows) * 100.0
                logger.info(
                    f"Progress: {pct:5.1f}% | RG: {processed_rgs}/{total_rgs} | "
                    f"Rows: {processed_rows:,}/{total_est_rows:,} | "
                    f"Symbols: {len(symbols_written)} | "
                    f"Speed: {rate / 1e6:.2f}M rows/s | "
                    f"ETA: {int(eta_s // 60)}m {int(eta_s % 60):02d}s"
                )

    # Flush final symbol
    if current_sym is not None and current_chunks:
        flush_symbol(current_sym, current_chunks, OUTPUT_DIR)
        symbols_written.add(current_sym)

    elapsed_total = time.time() - t_start
    final_files = list(OUTPUT_DIR.glob("*.parquet"))
    total_size_mb = sum(f.stat().st_size for f in final_files) / (1024 * 1024)

    logger.info("=" * 60)
    logger.info(f"✅ Partitioning Complete in {elapsed_total:.1f}s ({elapsed_total / 60:.1f} mins)!")
    logger.info(f"Total symbols partitioned: {len(final_files)} files")
    logger.info(f"Total storage on disk: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_partitioning()
