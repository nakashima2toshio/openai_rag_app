# embeddings: streamlit run a3_02_rag_embeddings_processing_technique.py --server.port=8506
# 基本動作の前に、embedding, vector化： datasetsの作成
# [RAG] Menu(Embeddings) ----------------------------
# (1) Embedding 取得・基本動作確認: embedding_demo
#                              : embedding_basic_01
# (1-1) Embedding 取得・基本動作確認-B
# (2) 文章検索 (Similarity Search)
# (3) コード検索
# (4) レコメンデーションシステム
# (5) Embedding の次元削減・正規化
# (6) 質問応答 (QA) システムへの Embeddings 活用
# (7) 可視化 (t-SNEなど) とクラスタリング
# (8) 機械学習モデルでの回帰・分類タスク
# (9) ゼロショット分類
# --------------------------------------------
# [datasets] ユースケース（Amazon の高級食品レビュー データセット）
# https://platform.openai.com/docs/guides/embeddings#use-cases
# ---------------------------------------------
# Streamlit アプリ: OpenAI Embeddings Playground
# 起動例: `streamlit run a3_02_rag_embeddings_processing_technique.py --server.port 8506`

import sys

import json
import tempfile
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# --- プロジェクト直下を import パスに追加 ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# --- helper ---
from a0_common_helper.helper import (
    init_page,
    init_messages,
    select_model,
    sanitize_key,
    get_default_messages,
    extract_text_from_response, append_user_message,
)
from utils.get_embedding import get_embedding
from utils.get_data_df import get_data_df

# --------------------------------------------------
# Streamlit ページ初期化
# --------------------------------------------------
st.set_page_config(page_title="OpenAI Embeddings App", page_icon="🔍")
init_page("OpenAI Embeddings App")

def select_model() -> str:
    emb_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    return st.sidebar.radio("Embeddingモデルを選択", emb_models, key="selected_emb_model")

# --------------------------------------------------
# 0. Embedding Demo
# --------------------------------------------------
def embedding_demo(_: str = "Embedding Demo") -> None:
    st.header("Embedding Demo")
    model = select_model()
    st.markdown(f"**選択モデル:** `{model}`")

    with st.form("embedding_form", clear_on_submit=False):
        user_text = st.text_area("テキストを入力してください", height=120)
        ok = st.form_submit_button("Embedding 取得")

    if ok and user_text.strip():
        vector = get_embedding(user_text, model=model)
        st.write("埋め込みベクトル（長さ =", len(vector), "）:")
        st.write(vector[:10], "...")  # 先頭10要素だけ表示


# --------------------------------------------------
# 1. Embedding 基本動作確認 – 単発入力
# --------------------------------------------------
def embedding_text_embedding(_: str = "Embedding Basic 01") -> None:
    model_name = select_model()
    st.subheader("単発テキスト → Embedding")

    if "basic_input" not in st.session_state:
        st.session_state.basic_input = ""

    with st.form("basic_form"):
        st.session_state.basic_input = st.text_area(
            "ここにテキストを入力してください",
            value=st.session_state.basic_input,
            height=120,
        )
        ok = st.form_submit_button("送信")

    if ok and st.session_state.basic_input.strip():
        vector = get_embedding(st.session_state.basic_input, model=model_name)
        st.write("ベクトル長 =", len(vector))
        st.json(vector[:20])  # 先頭20要素を簡易表示

# --------------------------------------------------
# 1-a. JSON → Embedding Vector Store 作成 Demo
# --------------------------------------------------
def embedding_01_a(_: str) -> None:
    model = select_model()
    st.subheader("JSON → Vector Store 作成デモ")
    st.info(f"使用モデル: `{model}`")

    data, df = get_data_df("utils/agents_docs.json")
    if df.empty:
        st.error("JSON ファイルが見つからないか空です。")
        return

    client = OpenAI()

    for idx, row in df.iterrows():
        key, raw_val = row["key"], row["value"]
        text = raw_val if isinstance(raw_val, str) else json.dumps(raw_val, ensure_ascii=False)

        with st.expander(f"{idx+1}/{len(df)}: {key}", expanded=False):
            st.code(text[:300] + ("…" if len(text) > 300 else ""), language="")

            # Embedding 取得
            vec = get_embedding(text, model=model)
            st.write("Embedding length:", len(vec))

            # JSONL を一時ファイル化
            with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
                tmp.write(json.dumps({"id": key, "text": text}, ensure_ascii=False) + "\n")
                tmp_path = tmp.name

            # OpenAI ファイルアップロード & Vector Store 作成
            uploaded = client.files.create(file=open(tmp_path, "rb"), purpose="user_data")
            vs = client.vector_stores.create(name=key)
            client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded.id)
            st.success(f"VectorStore `{vs.id}` を作成 (file_id={uploaded.id})")


# --------------------------------------------------
# コサイン類似度 & 文章検索
# --------------------------------------------------

# ① どこかで DataFrame を作った直後に埋め込み列をパース
def ensure_vector_col(df: pd.DataFrame, col: str = "embedding") -> pd.DataFrame:
    """embedding 列が str の場合は JSON でパースして list[float] に直す"""
    def _to_vec(x: Union[str, list, np.ndarray]) -> List[float]:
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, str):
            # "[0.12, 0.34, ...]" 形式 or '["0.12", "0.34"]'
            return json.loads(x)
        raise TypeError(f"Unsupported type for embedding: {type(x)}")
    df[col] = df[col].apply(_to_vec)
    return df


# ② コサイン類似度 (型チェック + 0-除算ガード込み)
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


# ③ 類似検索時に「列保証 → 類似度計算」
def similarity_search(df: pd.DataFrame, query_vec: List[float], top_k: int = 3) -> pd.DataFrame:
    df = ensure_vector_col(df)            # ←★ここで str→list を確実に
    df = df.copy()
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(query_vec, x))
    return df.nlargest(top_k, "similarity")

# -------------------------------------------
#
# -------------------------------------------
def embedding_02(_: str) -> None:
    st.subheader("文章検索 (Similarity Search)")

    model = select_model()
    st.markdown("**テスト用サンプルデータ生成**")
    sample_texts = [
        "おいしいコーヒーのお店です。",
        "格安で犬の餌を買えるサイト。",
        "最新のAI技術を解説する記事です。",
    ]
    df = pd.DataFrame({"text": sample_texts})
    df["embedding"] = df["text"].apply(lambda t: get_embedding(t, model=model))
    st.write("サンプル DataFrame:")
    st.dataframe(df[["text"]])

    query = st.text_input("検索クエリを入力", value="AI技術について知りたい")
    if st.button("検索実行"):
        top_df = similarity_search(df, query, model=model, top_k=3)
        st.dataframe(top_df[["text", "similarity"]])


# --------------------------------------------------
# ダミー枠（後日実装予定）
# --------------------------------------------------
def embedding_03(_: str): st.info("未実装: コード検索")
def embedding_04(_: str): st.info("未実装: レコメンデーション")
def embedding_05(_: str): st.info("未実装: 次元削減・正規化")
def embedding_06(_: str): st.info("未実装: QAシステムへの応用")
def embedding_07(_: str): st.info("未実装: 可視化 / クラスタリング")
def embedding_08(_: str): st.info("未実装: 回帰・分類タスク")
def embedding_09(_: str): st.info("未実装: ゼロショット分類")


# --------------------------------------------------
# サイドバー & ルーティング
# --------------------------------------------------
DEMO_FUNCS = {
    "(demo) Embedding Demo": embedding_demo,
    "(1) 入力テキストEmbedding": embedding_text_embedding,
    "  1-a JSON → VectorStore": embedding_01_a,
    "(2) 文章検索 (Similarity Search)": embedding_02,
    "(3) コード検索": embedding_03,
    "(4) レコメンデーション": embedding_04,
    "(5) 次元削減・正規化": embedding_05,
    "(6) QAシステム応用": embedding_06,
    "(7) 可視化 & クラスタリング": embedding_07,
    "(8) 回帰・分類タスク": embedding_08,
    "(9) ゼロショット分類": embedding_09,
}


def main() -> None:
    init_messages("embedding")
    choice = st.sidebar.radio("デモを選択してください", list(DEMO_FUNCS.keys()))
    DEMO_FUNCS[choice](choice)


if __name__ == "__main__":
    main()

# streamlit run a3_02_rag_embeddings_processing_technique.py
