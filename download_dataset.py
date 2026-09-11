"""
Automated dataset downloader for vaibhavs28/Dataset.
Downloads the compiled DuckDB database (market_data.duckdb) containing 11.65M+
75-minute candles and 3.56M+ daily candles directly from GitHub Release v1.0-data
into data/market_data.duckdb with progress display.
"""

import os
import sys
import time
from pathlib import Path
import requests
import urllib3

urllib3.disable_warnings()

RELEASE_URL = "https://github.com/vaibhavs28/Dataset/releases/download/v1.0-data/market_data.duckdb"
DEST_PATH = Path(__file__).resolve().parent / "data" / "market_data.duckdb"

def download():
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Target: {DEST_PATH}")
    print(f"Source: {RELEASE_URL}")

    if DEST_PATH.exists() and DEST_PATH.stat().st_size > 900 * 1024 * 1024:
        print(f"✓ Dataset already exists ({DEST_PATH.stat().st_size / (1024*1024):.1f} MB).")
        if "--force" not in sys.argv:
            print("To re-download, pass --force")
            return

    temp_path = DEST_PATH.with_suffix(".tmp")
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming download from {existing_size / (1024*1024):.1f} MB...")

    response = requests.get(RELEASE_URL, headers=headers, stream=True, verify=False, timeout=60)
    if response.status_code not in (200, 206):
        print(f"Error downloading: HTTP {response.status_code}")
        sys.exit(1)

    total_size = int(response.headers.get("content-length", 0)) + existing_size
    downloaded = existing_size
    start_time = time.time()
    last_print = 0

    mode = "ab" if existing_size > 0 else "wb"
    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2:
                    pct = (downloaded / total_size * 100) if total_size else 0
                    speed = ((downloaded - existing_size) / (1024 * 1024)) / (now - start_time) if (now - start_time) > 0 else 0
                    print(f"\rDownloading: {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s", end="", flush=True)
                    last_print = now

    print(f"\nDownload complete! Renaming {temp_path.name} -> {DEST_PATH.name}")
    temp_path.replace(DEST_PATH)
    print(f"✓ Successfully saved {DEST_PATH} ({DEST_PATH.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    download()
