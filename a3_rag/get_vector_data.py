# get_vector_data.py
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ----------------------------------------
# 1. 設定
# ----------------------------------------
# DATA_DIR = Path("data")
DATA_DIR = Path(".")
DATA_DIR.mkdir(exist_ok=True)

datasets_to_download = [
    {
        "name": "medical_qa",
        "hfpath": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "config": "en",
        "split": "train",
    },
    {
        "name": "legal_qa",
        "hfpath": "nguha/legalbench",
        "config": "consumer_contracts_qa",   # ★必須
        "split": "train",
    },
    {
        "name": "trivia_qa",
        "hfpath": "trivia_qa",
        "config": "rc",
        "split": "train",
    },
    {
        "name": "sciq_qa",
        "hfpath": "sciq",
        "config": None,
        "split": "train",
    },
    {
        "name": "customer_support_faq",
        "hfpath": "MakTek/Customer_support_faqs_dataset",
        "config": None,
        "split": "train",
    },
]

# ----------------------------------------
# 2. ダウンロード & 保存
# ----------------------------------------
def main():
    for d in datasets_to_download:
        print(f"▼ downloading {d['name']} …")
        ds = load_dataset(
            path=d["hfpath"],
            name=d["config"],
            split=d["split"],
        )

        # Arrow 形式 → data/<name>
        # arrow_path = DATA_DIR / d["name"]
        # ds.save_to_disk(arrow_path)
        # print(f"  saved dataset ➜ {arrow_path}")

        # CSV 形式 → data/<name>.csv
        csv_path = DATA_DIR / f"{d['name']}.csv"
        ds.to_pandas().to_csv(csv_path, index=False)
        print(f"  saved CSV     ➜ {csv_path}")

    print("\n All datasets downloaded & saved.")

if __name__ == "__main__":
    main()
