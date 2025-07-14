# python a30_00_rag_dataset_from_huggingface.py
# --------------------------------------------------
# ① カスタマーサポート・FAQデータセット   推奨データセット： Amazon_Polarity
# ② 一般知識・トリビアQAデータセット      推奨データセット： trivia_qa
# ③ 医療質問回答データセット             推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
# ④ 科学・技術QAデータセット             推奨データセット： sciq
# ⑤ 法律・判例QAデータセット             推奨データセット： nguha/legalbench
# --------------------------------------------------
import os
import re
import time
from pathlib import Path
from typing import List

from datasets import load_dataset
import pandas as pd, tempfile, textwrap
from pydantic import BaseModel, Field
from tqdm import tqdm

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent       # Paslib
ROOT_DIR = Path(__file__).resolve().parent

datasets_to_download = [
    {
        "name": "customer_support_faq",
        "hfpath": "MakTek/Customer_support_faqs_dataset",
        "config": None,
        "split": "train",
    },
    {
        "name": "trivia_qa",
        "hfpath": "trivia_qa",
        "config": "rc",
        "split": "train",
    },
    {
        "name": "medical_qa",
        "hfpath": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "config": "en",
        "split": "train",
    },
    {
        "name": "sciq_qa",
        "hfpath": "sciq",
        "config": None,
        "split": "train",
    },
    {
        "name": "legal_qa",
        "hfpath": "nguha/legalbench",
        "config": "consumer_contracts_qa",  # ★必須
        "split": "train",
    },
]

def download_dataset():
    DATA_DIR = Path("datasets")
    DATA_DIR.mkdir(exist_ok=True)

    for d in datasets_to_download:
        print(f"▼ downloading {d['name']} …")
        ds = load_dataset(
            path=d["hfpath"],
            name=d["config"],
            split=d["split"],
        )

        # # Arrow 形式 → data/<name>
        # arrow_path = DATA_DIR / d["name"]
        # ds.save_to_disk(arrow_path)
        # print(f"  saved dataset ➜ {arrow_path}")

        # CSV 形式 → data/<name>.csv
        csv_path = DATA_DIR / f"{d['name']}.csv"
        ds.to_pandas().to_csv(csv_path, index=False)
        print(f"  saved CSV     ➜ {csv_path}")

    print("\n[OK] All datasets downloaded & saved.")

def show_dataset():
    DATASETS_DIR = ROOT_DIR / "datasets"

    # ① カスタマーサポート・FAQデータセット 推奨データセット： Amazon_Polarity
    csv_path = DATASETS_DIR / "customer_support_faq.csv"
    df = pd.read_csv(csv_path)
    print(df.head(10))

    # ② 一般知識・トリビアQAデータセット      推奨データセット： trivia_qa
    # ③ 医療質問回答データセット             推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
    # ④ 科学・技術QAデータセット             推奨データセット： sciq
    # ⑤ 法律・判例QAデータセット             推奨データセット： nguha/legalbench

def main():
    show_dataset()

if __name__ == "__main__":
    main()
