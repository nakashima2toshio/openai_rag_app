# python a20_21_file_searches.py --server.port=8504
# 依存: pip install openai pandas tqdm
import time
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# --------------------------------------------------
# datasets フォルダを動的決定
# FAQのデータセットだけ、QA形式に変換する。
# --------------------------------------------------
def resolve_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    if list(here.glob("*.csv")):               # スクリプト直下に CSV
        return here
    if (here / "datasets").exists():           # サブフォルダ datasets/
        return here / "datasets"
    raise FileNotFoundError("CSV または datasets/ が見当たりません。")


DATA_DIR: Path = resolve_data_dir()
MAPPING_JSON: Path = DATA_DIR / "vector_store_mapping.json"


# --------------------------------------------------
# CSV 列挙
# --------------------------------------------------
def conv_qa_format(data_dir: Path = DATA_DIR) -> list[Path]:
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        print("⚠️  CSV が見つかりません:", data_dir)
    else:
        print("📄  検出した CSV:")
        for p in csv_files:
            print(f"  - {p.relative_to(data_dir)}")
    return csv_files


# --------------------------------------------------
# 列名検出ユーティリティ
# --------------------------------------------------
def detect_column(df: pd.DataFrame, candidates: set[str]) -> str | None:
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col
    return None


# --------------------------------------------------
# CSV → QA プレーンテキスト変換
# --------------------------------------------------
def set_qa_format_to_dataset(csv_path: Path) -> Path | None:
    df = pd.read_csv(csv_path)

    question_aliases = {"question", "questions", "prompt", "query", "problem", "ques"}
    answer_aliases = {"answer", "answers", "response", "solution", "ans"}

    q_col = detect_column(df, question_aliases)
    a_col = detect_column(df, answer_aliases)

    if q_col is None or a_col is None:
        print(f"⚠️  必要列を特定できずスキップ: {csv_path.name}  cols={list(df.columns)}")
        return None

    out_path = csv_path.with_name(f"qa_{csv_path.stem}.txt")
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            q = str(row[q_col]).strip()
            a = str(row[a_col]).strip()
            f.write(f"Q: {q}\nA: {a}\n\n")

    print("📝  Plain-text saved:", out_path.relative_to(DATA_DIR))
    return out_path


# --------------------------------------------------
# Vector Store 作成 & ファイルアップロード
# --------------------------------------------------
def create_vector_store_and_upload(txt_path: Path) -> str:
    client = OpenAI()
    vs = client.vector_stores.create(name=txt_path.name)
    vs_id = vs.id

    with txt_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
    client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_obj.id)

    while client.vector_stores.retrieve(vs_id).status != "completed":
        time.sleep(2)

    print("✅  Vector Store ready:", vs_id)
    return vs_id


# --------------------------------------------------
# メインワークフロー
# --------------------------------------------------
def make_dataset() -> dict[str, str]:
    mapping: dict[str, str] = {}
    csv_files = conv_qa_format()

    for csv_path in tqdm(csv_files, desc="Vectorising"):
        txt_path = set_qa_format_to_dataset(csv_path)
        if txt_path is None:                   # 列不足でスキップ
            continue
        vs_id = create_vector_store_and_upload(txt_path)
        mapping[csv_path.name] = vs_id

    if not mapping:
        print("❌  Vector Store が 1 つも生成されませんでした。")
        return mapping

    MAPPING_JSON.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_JSON.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    print("💾  Mapping saved to", MAPPING_JSON)
    return mapping


# --------------------------------------------------
# 検索デモ（単一 VS）
# --------------------------------------------------
def standalone_search(vs_id: str, query: str = "返品は何日以内？"):
    client = OpenAI()
    results = client.vector_stores.search(vector_store_id=vs_id, query=query)
    for r in results.data:
        print(f"{r.score:.3f}", r.content[0].text.strip()[:60], "...")


# --------------------------------------------------
# 検索デモ（マッピングファイルを利用）
# --------------------------------------------------
def file_searches():
    if not MAPPING_JSON.exists():
        print("❌  先に make_dataset() を実行してください。")
        return
    mapping = json.loads(MAPPING_JSON.read_text())
    for fname, vs_id in mapping.items():
        print(f"\n🔎  {fname}:")
        standalone_search(vs_id)


# --------------------------------------------------
# エントリポイント
# --------------------------------------------------
def main():
    make_dataset()          # Vector Store 作成
    # file_searches()       # 検索を試す場合はコメント解除


if __name__ == "__main__":
    main()

