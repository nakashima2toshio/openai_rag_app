# helper_rag.py
# RAGデータ前処理の共通機能
# -----------------------------------------
import re
import io
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# 既存のヘルパーモジュールをインポート
try:
    from helper_api import (
        config, logger, safe_json_dumps,
        format_timestamp, sanitize_key
    )
    from helper_st import (
        UIHelper, SessionStateManager, error_handler_ui
    )
except ImportError as e:
    # フォールバック用のログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"ヘルパーモジュールのインポートに失敗しました: {e}")


# ==================================================
# RAGデータ前処理の共通設定
# ==================================================
class RAGConfig:
    """RAG前処理の設定管理"""

    # データセット別の設定
    DATASET_CONFIGS = {
        "medical_qa"          : {
            "name"            : "医療QAデータ",
            "icon"            : "🏥",
            "required_columns": ["Question", "Complex_CoT", "Response"],
            "description"     : "医療質問回答データセット",
            "combine_template": "{question} {complex_cot} {response}"
        },
        "customer_support_faq": {
            "name"            : "カスタマーサポート・FAQ",
            "icon"            : "💬",
            "required_columns": ["question", "answer"],  # 要確認
            "description"     : "カスタマーサポートFAQデータセット",
            "combine_template": "{question} {answer}"
        },
        "legal_qa"            : {
            "name"            : "法律・判例QA",
            "icon"            : "⚖️",
            "required_columns": ["question", "answer"],  # 要確認
            "description"     : "法律・判例質問回答データセット",
            "combine_template": "{question} {answer}"
        },
        "sciq_qa"             : {
            "name"            : "科学・技術QA",
            "icon"            : "🔬",
            "required_columns": ["question", "correct_answer"],  # 要確認
            "description"     : "科学・技術質問回答データセット",
            "combine_template": "{question} {correct_answer}"
        },
        "trivia_qa"           : {
            "name"            : "一般知識・トリビアQA",
            "icon"            : "🧠",
            "required_columns": ["question", "answer"],  # 要確認
            "description"     : "一般知識・トリビア質問回答データセット",
            "combine_template": "{question} {answer}"
        }
    }

    @classmethod
    def get_config(cls, dataset_type: str) -> Dict[str, Any]:
        """データセット設定の取得"""
        return cls.DATASET_CONFIGS.get(dataset_type, {
            "name"            : "未知のデータセット",
            "icon"            : "❓",
            "required_columns": [],
            "description"     : "未知のデータセット",
            "combine_template": "{}"
        })


# ==================================================
# テキスト前処理の共通関数
# ==================================================
def clean_text(text: str) -> str:
    """
    テキストのクレンジング処理
    - 改行の除去
    - 連続した空白を1つの空白にまとめる
    - 不要な文字の正規化

    Args:
        text (str): 処理対象のテキスト

    Returns:
        str: クレンジング済みテキスト
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


def combine_columns(
        row: pd.Series,
        dataset_type: str = "medical_qa",
        custom_template: str = None
) -> str:
    """
    複数列を結合して1つのテキストにする（Vector Store/RAG用に最適化）

    Args:
        row (pd.Series): DataFrameの1行データ
        dataset_type (str): データセットタイプ
        custom_template (str, optional): カスタム結合テンプレート

    Returns:
        str: 結合済みテキスト
    """
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]
    template = custom_template or config_data["combine_template"]

    # 各列からテキストを抽出・クレンジング
    cleaned_values = {}
    for col in required_columns:
        value = row.get(col, '')
        cleaned_values[col.lower()] = clean_text(str(value))

    try:
        # テンプレートを使用して結合
        if dataset_type == "medical_qa":
            # 医療QA用の特別処理
            question = cleaned_values.get('question', '')
            complex_cot = cleaned_values.get('complex_cot', '')
            response = cleaned_values.get('response', '')
            combined = f"{question} {complex_cot} {response}"
        else:
            # 他のデータセット用の汎用処理
            combined = template.format(**cleaned_values)
    except KeyError as e:
        logger.warning(f"テンプレート処理エラー: {e}")
        # フォールバック: 全ての値を空白区切りで結合
        combined = " ".join(cleaned_values.values())

    return combined.strip()


def additional_preprocessing(df: pd.DataFrame, dataset_type: str = None) -> pd.DataFrame:
    """
    その他の前処理
    - 重複行の除去
    - 空行の除去
    - インデックスのリセット

    Args:
        df (pd.DataFrame): 前処理対象のDataFrame
        dataset_type (str, optional): データセットタイプ

    Returns:
        pd.DataFrame: 前処理済みDataFrame
    """
    initial_rows = len(df)

    # 重複行の除去
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    # データセット別の必須列を取得
    if dataset_type:
        config_data = RAGConfig.get_config(dataset_type)
        required_columns = config_data["required_columns"]
        # 存在する必須列のみでフィルタ
        existing_required = [col for col in required_columns if col in df.columns]
        if existing_required:
            df = df.dropna(subset=existing_required)
    else:
        # 汎用処理: 全列がNAの行を除去
        df = df.dropna(how='all')

    empty_rows_removed = initial_rows - duplicates_removed - len(df)

    # インデックスのリセット
    df = df.reset_index(drop=True)

    # 処理統計をログ出力
    logger.info(f"前処理完了: 重複除去={duplicates_removed}行, 空行除去={empty_rows_removed}行")

    return df


# ==================================================
# データ検証の共通関数
# ==================================================
def validate_data(df: pd.DataFrame, dataset_type: str = None) -> List[str]:
    """
    データの検証

    Args:
        df (pd.DataFrame): 検証対象のDataFrame
        dataset_type (str, optional): データセットタイプ

    Returns:
        List[str]: 検証結果・問題点のリスト
    """
    issues = []

    # データセット設定の取得
    if dataset_type:
        config_data = RAGConfig.get_config(dataset_type)
        required_columns = config_data["required_columns"]
    else:
        required_columns = []

    # 基本統計
    issues.append(f"総行数: {len(df)}")
    issues.append(f"総列数: {len(df.columns)}")

    # 必須列の存在確認
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"⚠️ 必須列が不足: {missing_columns}")
        else:
            issues.append(f"✅ 必須列確認済み: {required_columns}")

    # 各列の空値確認
    for col in df.columns:
        empty_count = df[col].isna().sum() + (df[col] == '').sum()
        if empty_count > 0:
            percentage = (empty_count / len(df)) * 100
            issues.append(f"{col}列: 空値 {empty_count}個 ({percentage:.1f}%)")

    # 文字数の確認（必須列のみ）
    for col in required_columns:
        if col in df.columns:
            text_lengths = df[col].astype(str).str.len()
            avg_length = text_lengths.mean()
            max_length = text_lengths.max()
            min_length = text_lengths.min()
            issues.append(f"{col}列: 平均{avg_length:.1f}文字 (最小{min_length}, 最大{max_length})")

    # 重複行の確認
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"⚠️ 重複行: {duplicate_count}個")
    else:
        issues.append("✅ 重複行なし")

    return issues


# ==================================================
# ファイル処理の共通関数
# ==================================================
def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    データセットの読み込みと基本検証

    Args:
        uploaded_file: Streamlitのアップロードファイル
        dataset_type (str, optional): データセットタイプ

    Returns:
        Tuple[pd.DataFrame, List[str]]: データフレームと検証結果
    """
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(uploaded_file)

        # 基本検証
        validation_results = validate_data(df, dataset_type)

        logger.info(f"データセット読み込み完了: {len(df)}行, {len(df.columns)}列")
        return df, validation_results

    except Exception as e:
        logger.error(f"データセット読み込みエラー: {e}")
        raise


def create_download_data(df: pd.DataFrame,
                         include_combined: bool = True,
                         dataset_type: str = None) -> Tuple[str, Optional[str]]:
    """
    ダウンロード用データの作成

    Args:
        df (pd.DataFrame): 処理済みDataFrame
        include_combined (bool): 結合テキストを含めるか
        dataset_type (str, optional): データセットタイプ

    Returns:
        Tuple[str, Optional[str]]: CSVデータ、結合テキストデータ
    """
    try:
        logger.info(f"create_download_data開始: 行数={len(df)}, include_combined={include_combined}")

        # CSVデータの作成
        logger.info("CSVデータ作成開始")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()
        logger.info(f"CSVデータ作成完了: 長さ={len(csv_data)}")

        # 結合テキストデータの作成
        text_data = None
        if include_combined and 'Combined_Text' in df.columns:
            logger.info("結合テキストデータ作成開始")
            text_data = df['Combined_Text'].to_string(index=False)
            logger.info(f"結合テキストデータ作成完了: 長さ={len(text_data)}")
        else:
            if include_combined:
                logger.warning("Combined_Text列が見つかりません")
            else:
                logger.info("結合テキスト作成はスキップされました")

        logger.info("create_download_data完了")
        return csv_data, text_data

    except Exception as e:
        logger.error(f"create_download_data エラー: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        raise


# ==================================================
# 統計情報表示の共通関数
# ==================================================
def display_statistics(df_original: pd.DataFrame,
                       df_processed: pd.DataFrame,
                       dataset_type: str = None) -> None:
    """
    処理前後の統計情報を表示

    Args:
        df_original (pd.DataFrame): 元のDataFrame
        df_processed (pd.DataFrame): 処理後のDataFrame
        dataset_type (str, optional): データセットタイプ
    """
    st.subheader("📊 統計情報")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("元の行数", len(df_original))
    with col2:
        st.metric("処理後の行数", len(df_processed))
    with col3:
        removed_rows = len(df_original) - len(df_processed)
        st.metric("除去された行数", removed_rows)

    # 結合テキストの分析
    if 'Combined_Text' in df_processed.columns:
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
        try:
            import pandas as pd  # 最初にインポート

            # 文字数を適切な範囲でビン分け
            min_length = int(text_lengths.min())
            max_length = int(text_lengths.max())

            # ビン数を調整（最大20個）
            num_bins = min(20, max(5, (max_length - min_length) // 100))

            if num_bins >= 2 and max_length > min_length:
                # 等間隔でビンを作成
                bin_edges = pd.cut(text_lengths, bins=num_bins, duplicates='drop')
                bin_counts = bin_edges.value_counts().sort_index()

                # ビンのラベルを数値に変換
                bin_data = {}
                for interval, count in bin_counts.items():
                    # 区間の中央値をラベルとして使用
                    mid_point = int((interval.left + interval.right) / 2)
                    bin_data[f"{mid_point}文字"] = count

                # データフレームとして表示
                if bin_data:
                    chart_df = pd.DataFrame.from_dict(bin_data, orient='index', columns=['件数'])
                    st.bar_chart(chart_df)
                else:
                    st.info("文字数分布データが不足しています")
            else:
                st.info("文字数の範囲が狭く、ヒストグラムを表示できません")

        except Exception as e:
            logger.warning(f"ヒストグラム表示エラー: {e}")
            # フォールバック: 簡単な統計表示
            try:
                st.write("**文字数分布（簡易版）**")
                percentiles = text_lengths.quantile([0.25, 0.5, 0.75]).round(0)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("25%点", f"{percentiles[0.25]:.0f}文字")
                with col2:
                    st.metric("中央値", f"{percentiles[0.5]:.0f}文字")
                with col3:
                    st.metric("75%点", f"{percentiles[0.75]:.0f}文字")
            except Exception as e2:
                logger.error(f"フォールバック表示もエラー: {e2}")
                st.info("文字数分布の表示をスキップしました")


# ==================================================
# UI設定の共通関数
# ==================================================
def setup_rag_page(dataset_type: str = "medical_qa") -> None:
    """
    RAGデータ前処理ページの共通設定

    Args:
        dataset_type (str): データセットタイプ
    """
    config_data = RAGConfig.get_config(dataset_type)

    st.set_page_config(
        page_title=f"{config_data['name']}前処理",
        page_icon=config_data['icon'],
        layout="wide"
    )

    st.title(f"{config_data['icon']} {config_data['name']}前処理アプリ")
    st.markdown("---")

    # データセット情報の表示
    with st.sidebar.expander("📋 データセット情報", expanded=True):
        st.write(f"**名前**: {config_data['name']}")
        st.write(f"**説明**: {config_data['description']}")
        st.write(f"**必須列**: {', '.join(config_data['required_columns'])}")


def setup_sidebar_controls(dataset_type: str = "medical_qa") -> Tuple[bool, bool]:
    """
    サイドバーの制御パネル設定

    Args:
        dataset_type (str): データセットタイプ

    Returns:
        Tuple[bool, bool]: (列結合オプション, データ検証表示オプション)
    """
    st.sidebar.header("設定")

    combine_columns_option = st.sidebar.checkbox(
        "複数列を結合する（Vector Store用）",
        value=True,
        help="複数列を結合してRAG用テキストを作成"
    )

    show_validation = st.sidebar.checkbox(
        "データ検証を表示",
        value=True,
        help="データの品質検証結果を表示"
    )

    return combine_columns_option, show_validation


# ==================================================
# エラーハンドリング付きの処理関数
# ==================================================
# 元の関数（スピナーあり）:
# @error_handler_ui
# def process_rag_data(df: pd.DataFrame,
#                      dataset_type: str,
#                      combine_columns_option: bool = True) -> pd.DataFrame:
@error_handler_ui
def process_rag_data(df: pd.DataFrame,
                     dataset_type: str,
                     combine_columns_option: bool = True) -> pd.DataFrame:
    """
    RAGデータの前処理を実行

    Args:
        df (pd.DataFrame): 元のDataFrame
        dataset_type (str): データセットタイプ
        combine_columns_option (bool): 列結合オプション

    Returns:
        pd.DataFrame: 処理済みDataFrame
    """
    # with st.spinner("前処理中..."):  # ← この行を削除
    # 基本的な前処理
    df_processed = additional_preprocessing(df.copy(), dataset_type)

    # 各列のクレンジング
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    for col in required_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(clean_text)

    # 列の結合（オプション）
    if combine_columns_option:
        df_processed['Combined_Text'] = df_processed.apply(
            lambda row: combine_columns(row, dataset_type),
            axis=1
        )

    return df_processed


# ==================================================
# 使用方法説明の共通関数
# ==================================================
def show_usage_instructions(dataset_type: str = "medical_qa") -> None:
    """
    使用方法の説明を表示

    Args:
        dataset_type (str): データセットタイプ
    """
    config_data = RAGConfig.get_config(dataset_type)
    required_columns_str = ", ".join(config_data["required_columns"])

    st.markdown("---")
    st.subheader("📖 使用方法")
    st.markdown(f"""
    1. **CSVファイルをアップロード**: {required_columns_str} の列を含むCSVファイルを選択
    2. **前処理を実行**: 以下の処理が自動で実行されます：
       - 改行の除去
       - 連続した空白の統一
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    3. **複数列結合**: Vector Store/RAG用に最適化された自然な文章として結合
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


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    # 設定クラス
    'RAGConfig',

    # 前処理関数
    'clean_text',
    'combine_columns',
    'additional_preprocessing',

    # 検証・ファイル処理
    'validate_data',
    'load_dataset',
    'create_download_data',

    # UI関連
    'setup_rag_page',
    'setup_sidebar_controls',
    'display_statistics',
    'show_usage_instructions',

    # 処理関数
    'process_rag_data',
]
