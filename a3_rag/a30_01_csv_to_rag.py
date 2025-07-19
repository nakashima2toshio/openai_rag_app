# streamlit run a30_01_csv_to_rag.py --server.port=8505
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import time
import json

import openai
import tiktoken

import streamlit as st

# Streamlitページ設定
st.set_page_config(
    page_title="OpenAI Embedding API サンプル",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 OpenAI Embedding API - FAQ処理サンプル")

# サイドバーでAPI設定
st.sidebar.header("設定")
api_key = os.getenv("OPENAI_API_KEY")
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if api_key:
#     openai.api_key = api_key

# モデル選択
model_options = {
    "text-embedding-3-small": "text-embedding-3-small",
    "text-embedding-3-large": "text-embedding-3-large"
}
selected_model = st.sidebar.selectbox("埋め込みモデル", list(model_options.keys()))

# データ形式選択
data_format = st.sidebar.selectbox(
    "データ形式",
    ["質問のみ", "回答のみ", "質問+回答結合", "個別処理"]
)


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """テキストのトークン数をカウント"""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except:
        # フォールバック: 文字数 / 4 で概算
        return len(text) // 4


def preprocess_text(text: str) -> str:
    """テキストの前処理"""
    if pd.isna(text) or text is None:
        return ""

    # 基本的なクリーニング
    text = str(text).strip()

    # 連続する空白を1つにまとめる
    import re
    text = re.sub(r'\s+', ' ', text)

    # 特殊文字の正規化（必要に応じて）
    text = text.replace('\u00a0', ' ')  # non-breaking space

    return text


def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """テキストをトークン数制限に合わせて切り詰め"""
    if count_tokens(text) <= max_tokens:
        return text

    # 単純に文字数で切り詰め（より正確にはトークンレベルで切り詰めるべき）
    words = text.split()
    truncated = ""
    for word in words:
        test_text = truncated + " " + word if truncated else word
        if count_tokens(test_text) > max_tokens:
            break
        truncated = test_text

    return truncated


def prepare_embedding_data(df: pd.DataFrame, format_type: str) -> List[Dict]:
    """埋め込み用データの準備"""
    data = []

    for idx, row in df.iterrows():
        question = preprocess_text(row['question'])
        answer = preprocess_text(row['answer'])

        if format_type == "質問のみ":
            if question:
                data.append({
                    'id'      : f"q_{idx}",
                    'text'    : truncate_text(question),
                    'metadata': {'type': 'question', 'index': idx}
                })

        elif format_type == "回答のみ":
            if answer:
                data.append({
                    'id'      : f"a_{idx}",
                    'text'    : truncate_text(answer),
                    'metadata': {'type': 'answer', 'index': idx}
                })

        elif format_type == "質問+回答結合":
            if question and answer:
                combined_text = f"質問: {question}\n回答: {answer}"
                data.append({
                    'id'      : f"qa_{idx}",
                    'text'    : truncate_text(combined_text),
                    'metadata': {'type': 'qa_pair', 'index': idx}
                })

        elif format_type == "個別処理":
            if question:
                data.append({
                    'id'      : f"q_{idx}",
                    'text'    : truncate_text(question),
                    'metadata': {'type': 'question', 'index': idx}
                })
            if answer:
                data.append({
                    'id'      : f"a_{idx}",
                    'text'    : truncate_text(answer),
                    'metadata': {'type': 'answer', 'index': idx}
                })

    return data


def get_embeddings_batch(texts: List[str], model: str, batch_size: int = 20) -> List[List[float]]:
    """バッチで埋め込みベクトルを取得"""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = openai.embeddings.create(
                input=batch,
                model=model
            )

            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

            # レート制限対策
            time.sleep(0.1)

        except Exception as e:
            st.error(f"埋め込み取得エラー (バッチ {i // batch_size + 1}): {str(e)}")
            return []

    return embeddings


# メインコンテンツ
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 データ確認")

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("データサンプル:")
        st.dataframe(df.head())

        st.write(f"総レコード数: {len(df)}")

        # データ統計
        if 'question' in df.columns and 'answer' in df.columns:
            question_lengths = df['question'].fillna('').astype(str).apply(len)
            answer_lengths = df['answer'].fillna('').astype(str).apply(len)

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("質問平均文字数", f"{question_lengths.mean():.0f}")
                st.metric("質問最大文字数", f"{question_lengths.max()}")

            with col1_2:
                st.metric("回答平均文字数", f"{answer_lengths.mean():.0f}")
                st.metric("回答最大文字数", f"{answer_lengths.max()}")

with col2:
    st.header("🚀 埋め込み処理")

    if uploaded_file and api_key:
        # データ準備
        embedding_data = prepare_embedding_data(df, data_format)

        st.write(f"処理対象テキスト数: {len(embedding_data)}")

        # サンプル表示
        if embedding_data:
            st.write("処理対象サンプル:")
            sample_data = embedding_data[:3]
            for i, item in enumerate(sample_data):
                with st.expander(f"サンプル {i + 1} (ID: {item['id']})"):
                    st.write(f"**タイプ:** {item['metadata']['type']}")
                    st.write(f"**トークン数:** {count_tokens(item['text'])}")
                    st.write(f"**テキスト:** {item['text'][:200]}...")

        # 埋め込み実行
        if st.button("埋め込みベクトル取得開始", type="primary"):
            if not embedding_data:
                st.warning("処理対象のデータがありません。")
            else:
                with st.spinner("埋め込みベクトルを取得中..."):
                    texts = [item['text'] for item in embedding_data]

                    # プログレスバー
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    embeddings = []
                    batch_size = 20
                    total_batches = (len(texts) + batch_size - 1) // batch_size

                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(texts))
                        batch_texts = texts[start_idx:end_idx]

                        status_text.text(f"バッチ {batch_idx + 1}/{total_batches} 処理中...")

                        try:
                            response = openai.embeddings.create(
                                input=batch_texts,
                                model=model_options[selected_model]
                            )

                            batch_embeddings = [data.embedding for data in response.data]
                            embeddings.extend(batch_embeddings)

                            progress_bar.progress((batch_idx + 1) / total_batches)
                            time.sleep(0.1)  # レート制限対策

                        except Exception as e:
                            st.error(f"エラー: {str(e)}")
                            break

                    if len(embeddings) == len(texts):
                        st.success(f"✅ 埋め込みベクトル取得完了！({len(embeddings)}個)")

                        # 結果の保存
                        result_data = []
                        for i, (item, embedding) in enumerate(zip(embedding_data, embeddings)):
                            result_data.append({
                                'id'              : item['id'],
                                'text'            : item['text'],
                                'embedding_vector': embedding,
                                'metadata'        : item['metadata']
                            })

                        # ベクトルの統計情報
                        embedding_array = np.array(embeddings)
                        st.write(f"**ベクトル次元数:** {embedding_array.shape[1]}")
                        st.write(f"**ベクトル統計:**")
                        st.write(f"- 平均値: {embedding_array.mean():.6f}")
                        st.write(f"- 標準偏差: {embedding_array.std():.6f}")

                        # ダウンロード機能
                        if st.button("結果をダウンロード（JSON）"):
                            # 埋め込みベクトルをリストに変換してJSON保存可能にする
                            download_data = []
                            for item in result_data:
                                download_item = item.copy()
                                download_item['embedding_vector'] = download_item['embedding_vector']
                                download_data.append(download_item)

                            json_str = json.dumps(download_data, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="📥 結果ダウンロード",
                                data=json_str,
                                file_name=f"embeddings_{selected_model}_{data_format}.json",
                                mime="application/json"
                            )

# 使用方法の説明
st.header("📝 使用方法とベストプラクティス")

with st.expander("効果的な前処理について"):
    st.markdown("""
    ### 1. テキストクリーニング
    - 不要な空白や改行の除去
    - HTMLタグやマークダウンの処理
    - 特殊文字の正規化

    ### 2. トークン数制限
    - text-embedding-3-small/large: 最大8,191トークン
    - 長いテキストは適切に分割または要約

    ### 3. データ形式の選択
    - **質問のみ**: 質問による検索に最適
    - **回答のみ**: 回答内容による検索に最適  
    - **質問+回答結合**: コンテキスト豊富、ファイルサイズ大
    - **個別処理**: 柔軟性高、但し2倍のAPIコール
    """)

with st.expander("コスト最適化のヒント"):
    st.markdown("""
    ### API使用量削減
    - バッチ処理でリクエスト数を削減
    - 適切なモデル選択（small vs large）
    - 重複データの事前除去
    - キャッシュ機能の活用

    ### モデル選択指針
    - **text-embedding-3-small**: コスト重視、基本的な類似度検索
    - **text-embedding-3-large**: 精度重視、複雑な意味理解
    """)

# 注意事項
st.warning(
    "⚠️ 注意: 本サンプルは学習用です。本番環境では適切なエラーハンドリング、ログ機能、セキュリティ対策を実装してください。")