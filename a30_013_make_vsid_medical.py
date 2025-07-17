# streamlit run a30_013_make_vsid_medical.py --server.port=8503
import streamlit as st
import pandas as pd
import re
import io
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    テキストのクレンジング処理
    - 改行の除去
    - 連続した空白を1つの空白にまとめる
    - 不要な文字の正規化
    """
    if pd.isna(text) or text == "":
        return ""

    # 改行を空白に置換
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 連続した空白を1つの空白にまとめる
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を除去
    text = text.strip()

    # 引用符の正規化
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text


def combine_columns(row: pd.Series) -> str:
    """
    3列を結合して1つのテキストにする（Vector Store/RAG用に最適化）
    """
    question = clean_text(str(row.get('Question', '')))
    complex_cot = clean_text(str(row.get('Complex_CoT', '')))
    response = clean_text(str(row.get('Response', '')))

    # vector store/embedding用：自然な文章として結合
    combined = f"{question} {complex_cot} {response}"

    return combined


def additional_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    その他の前処理
    """
    # 重複行の除去
    df = df.drop_duplicates()

    # 空行の除去
    df = df.dropna(subset=['Question', 'Response'])

    # インデックスのリセット
    df = df.reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame) -> List[str]:
    """
    データの検証
    """
    issues = []

    # 必須列の存在確認
    required_columns = ['Question', 'Complex_CoT', 'Response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"必須列が不足しています: {missing_columns}")

    # 空の値の確認
    for col in required_columns:
        if col in df.columns:
            empty_count = df[col].isna().sum() + (df[col] == '').sum()
            if empty_count > 0:
                issues.append(f"{col}列に空の値が{empty_count}個あります")

    # 文字数の確認
    for col in required_columns:
        if col in df.columns:
            avg_length = df[col].astype(str).str.len().mean()
            issues.append(f"{col}の平均文字数: {avg_length:.1f}")

    return issues


def main():
    st.set_page_config(
        page_title="医療QAデータ前処理",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 医療QAデータ前処理アプリ")
    st.markdown("---")

    # サイドバーでの設定
    st.sidebar.header("設定")
    combine_columns_option = st.sidebar.checkbox("3列を結合する（Vector Store用）", value=True)
    show_validation = st.sidebar.checkbox("データ検証を表示", value=True)

    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロードしてください",
        type=['csv'],
        help="Question, Complex_CoT, Response の3列を含むCSVファイル"
    )

    if uploaded_file is not None:
        try:
            # CSVファイルの読み込み
            df = pd.read_csv(uploaded_file)

            st.success(f"ファイルが正常に読み込まれました。行数: {len(df)}")

            # 元データの表示
            st.subheader("📋 元データ")
            st.dataframe(df.head(10))

            # データ検証
            if show_validation:
                st.subheader("🔍 データ検証")
                validation_issues = validate_data(df)
                for issue in validation_issues:
                    st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            if st.button("前処理を実行"):
                with st.spinner("前処理中..."):
                    # 基本的な前処理
                    df_processed = additional_preprocessing(df.copy())

                    # 各列のクレンジング
                    for col in ['Question', 'Complex_CoT', 'Response']:
                        if col in df_processed.columns:
                            df_processed[col] = df_processed[col].apply(clean_text)

                    # 列の結合（オプション）
                    if combine_columns_option:
                        df_processed['Combined_Text'] = df_processed.apply(combine_columns, axis=1)

                    st.success("前処理が完了しました！")

                    # 前処理後のデータ表示
                    st.subheader("✅ 前処理後のデータ")
                    st.dataframe(df_processed.head(10))

                    # 統計情報
                    st.subheader("📊 統計情報")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("元の行数", len(df))
                    with col2:
                        st.metric("処理後の行数", len(df_processed))
                    with col3:
                        removed_rows = len(df) - len(df_processed)
                        st.metric("除去された行数", removed_rows)

                    # 文字数分析
                    if combine_columns_option and 'Combined_Text' in df_processed.columns:
                        st.subheader("📝 結合後テキスト分析")
                        text_lengths = df_processed['Combined_Text'].str.len()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("平均文字数", f"{text_lengths.mean():.0f}")
                        with col2:
                            st.metric("最大文字数", text_lengths.max())
                        with col3:
                            st.metric("最小文字数", text_lengths.min())

                        # 文字数分布のヒストグラム
                        st.bar_chart(text_lengths.value_counts().head(20))

                    # ダウンロード
                    st.subheader("💾 ダウンロード")

                    # CSVとして保存
                    csv_buffer = io.StringIO()
                    df_processed.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="前処理済みデータをダウンロード (CSV)",
                        data=csv_data,
                        file_name="preprocessed_medical_qa.csv",
                        mime="text/csv"
                    )

                    # 結合テキストのみのダウンロード（embeddingに適した形式）
                    if combine_columns_option and 'Combined_Text' in df_processed.columns:
                        text_only = df_processed['Combined_Text'].to_string(index=False)
                        st.download_button(
                            label="結合テキストのみダウンロード (TXT)",
                            data=text_only,
                            file_name="combined_medical_qa.txt",
                            mime="text/plain"
                        )

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

    # 使用方法の説明
    st.markdown("---")
    st.subheader("📖 使用方法")
    st.markdown("""
    1. **CSVファイルをアップロード**: Question, Complex_CoT, Response の3列を含むCSVファイルを選択
    2. **前処理を実行**: 以下の処理が自動で実行されます：
       - 改行の除去
       - 連続した空白の統一
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    3. **3列結合**: Vector Store/RAG用に最適化された自然な文章として結合
    4. **ダウンロード**: 前処理済みデータをCSV形式でダウンロード

    **Vector Store用最適化:**
    - 自然な文章として結合（ラベル文字列なし）
    - OpenAI embeddingモデルに最適化
    - 検索性能が向上

    **追加の前処理項目:**
    - 重複データの除去
    - 空値の処理
    - 文字エンコーディングの統一
    - 特殊文字の正規化
    - データ検証とレポート
    """)


if __name__ == "__main__":
    main()

# streamlit run a30_013_make_vsid_medical.py --server.port=8503
