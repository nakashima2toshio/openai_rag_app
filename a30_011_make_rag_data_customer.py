# a30_012_make_rag_data_customer_standalone.py
# カスタマーサポートFAQデータのRAG前処理（モデル選択機能付き・完全独立版）
# 外部ヘルパーモジュールへの依存なし
# streamlit run a30_012_make_rag_data_customer_standalone.py --server.port=8501

import streamlit as st
import pandas as pd
import re
import io
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from functools import wraps

# ===================================================================
# 完全独立版：外部ヘルパーモジュールへの依存なし
# ===================================================================

# 基本ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# 設定管理クラス
# ==================================================
class AppConfig:
    """アプリケーション設定（完全独立実装）"""

    # 利用可能なモデル
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini"
    ]

    DEFAULT_MODEL = "gpt-4o-mini"

    # モデル料金（1000トークンあたりのドル）
    MODEL_PRICING = {
        "gpt-4o"                   : {"input": 0.005, "output": 0.015},
        "gpt-4o-mini"              : {"input": 0.00015, "output": 0.0006},
        "gpt-4o-audio-preview"     : {"input": 0.01, "output": 0.02},
        "gpt-4o-mini-audio-preview": {"input": 0.00025, "output": 0.001},
        "gpt-4.1"                  : {"input": 0.0025, "output": 0.01},
        "gpt-4.1-mini"             : {"input": 0.0001, "output": 0.0004},
        "o1"                       : {"input": 0.015, "output": 0.06},
        "o1-mini"                  : {"input": 0.003, "output": 0.012},
        "o3"                       : {"input": 0.03, "output": 0.12},
        "o3-mini"                  : {"input": 0.006, "output": 0.024},
        "o4"                       : {"input": 0.05, "output": 0.20},
        "o4-mini"                  : {"input": 0.01, "output": 0.04},
    }

    # モデル制限
    MODEL_LIMITS = {
        "gpt-4o"                   : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini"              : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-audio-preview"     : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini-audio-preview": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1"                  : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1-mini"             : {"max_tokens": 128000, "max_output": 4096},
        "o1"                       : {"max_tokens": 128000, "max_output": 32768},
        "o1-mini"                  : {"max_tokens": 128000, "max_output": 65536},
        "o3"                       : {"max_tokens": 200000, "max_output": 100000},
        "o3-mini"                  : {"max_tokens": 200000, "max_output": 100000},
        "o4"                       : {"max_tokens": 256000, "max_output": 128000},
        "o4-mini"                  : {"max_tokens": 256000, "max_output": 128000},
    }

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(model, {"max_tokens": 128000, "max_output": 4096})

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.00015, "output": 0.0006})


# ==================================================
# RAG設定クラス
# ==================================================
class RAGConfig:
    """RAGデータ前処理の設定"""

    DATASET_CONFIGS = {
        "customer_support_faq": {
            "name"            : "カスタマーサポート・FAQ",
            "icon"            : "💬",
            "required_columns": ["question", "answer"],
            "description"     : "カスタマーサポートFAQデータセット",
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
# トークン管理クラス
# ==================================================
class TokenManager:
    """トークン数の管理（簡易版）"""

    @staticmethod
    def count_tokens(text: str, model: str = None) -> int:
        """テキストのトークン数をカウント（簡易推定）"""
        if not text:
            return 0

        # 簡易推定: 日本語文字は0.5トークン、英数字は0.25トークン
        japanese_chars = len([c for c in text if ord(c) > 127])
        english_chars = len(text) - japanese_chars
        estimated_tokens = int(japanese_chars * 0.5 + english_chars * 0.25)

        # 最低1トークンは必要
        return max(1, estimated_tokens)

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """API使用コストの推定"""
        pricing = AppConfig.get_model_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


# ==================================================
# エラーハンドリングデコレータ
# ==================================================
def safe_execute(func):
    """安全実行デコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"エラーが発生しました: {str(e)}")
            return None

    return wrapper


# ==================================================
# UI関数群
# ==================================================
def select_model(key: str = "model_selection") -> str:
    """モデル選択UI"""
    models = AppConfig.AVAILABLE_MODELS
    default_model = AppConfig.DEFAULT_MODEL

    try:
        default_index = models.index(default_model)
    except ValueError:
        default_index = 0

    selected = st.sidebar.selectbox(
        "🤖 モデルを選択",
        models,
        index=default_index,
        key=key,
        help="利用するOpenAIモデルを選択してください"
    )

    return selected


def show_model_info(selected_model: str) -> None:
    """選択されたモデルの情報を表示"""
    try:
        limits = AppConfig.get_model_limits(selected_model)
        pricing = AppConfig.get_model_pricing(selected_model)

        with st.sidebar.expander("📊 選択モデル情報", expanded=False):
            # 基本情報
            col1, col2 = st.columns(2)
            with col1:
                st.write("**最大入力**")
                st.write(f"{limits['max_tokens']:,}")
            with col2:
                st.write("**最大出力**")
                st.write(f"{limits['max_output']:,}")

            # 料金情報
            st.write("**料金（1000トークン）**")
            st.write(f"- 入力: ${pricing['input']:.5f}")
            st.write(f"- 出力: ${pricing['output']:.5f}")

            # モデル特性
            if selected_model.startswith("o"):
                st.info("🧠 推論特化モデル")
                st.caption("高度な推論タスクに最適化")
            elif "audio" in selected_model:
                st.info("🎵 音声対応モデル")
                st.caption("音声入力・出力に対応")
            elif "gpt-4o" in selected_model:
                st.info("👁️ マルチモーダルモデル")
                st.caption("テキスト・画像の理解が可能")
            else:
                st.info("💬 標準対話モデル")
                st.caption("一般的な対話・テキスト処理")

            # RAG用途での推奨度
            st.write("**RAG用途推奨度**")
            if selected_model in ["gpt-4o-mini", "gpt-4.1-mini"]:
                st.success("✅ 最適（コスト効率良好）")
            elif selected_model in ["gpt-4o", "gpt-4.1"]:
                st.info("💡 高品質（コスト高）")
            elif selected_model.startswith("o"):
                st.warning("⚠️ 推論特化（RAG用途には過剰）")
            else:
                st.info("💬 標準的な性能")

    except Exception as e:
        logger.error(f"モデル情報表示エラー: {e}")
        st.sidebar.error("モデル情報の取得に失敗しました")


def estimate_token_usage(df_processed: pd.DataFrame, selected_model: str) -> None:
    """処理済みデータのトークン使用量推定"""
    try:
        if 'Combined_Text' in df_processed.columns:
            # サンプルテキストでトークン数を推定
            sample_size = min(10, len(df_processed))
            sample_texts = df_processed['Combined_Text'].head(sample_size).tolist()
            total_chars = df_processed['Combined_Text'].str.len().sum()

            if sample_texts:
                sample_text = " ".join(sample_texts)
                sample_tokens = TokenManager.count_tokens(sample_text, selected_model)
                sample_chars = len(sample_text)

                if sample_chars > 0:
                    # 全体のトークン数を推定
                    estimated_total_tokens = int((total_chars / sample_chars) * sample_tokens)

                    with st.expander("🔢 トークン使用量推定", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("推定総トークン数", f"{estimated_total_tokens:,}")
                        with col2:
                            avg_tokens_per_record = estimated_total_tokens / len(df_processed)
                            st.metric("平均トークン/レコード", f"{avg_tokens_per_record:.0f}")
                        with col3:
                            # embedding用のコスト推定（参考値）
                            embedding_cost = (estimated_total_tokens / 1000) * 0.0001
                            st.metric("推定embedding費用", f"${embedding_cost:.4f}")

                        st.info(f"💡 選択モデル「{selected_model}」での推定値")
                        st.caption("※ 実際のトークン数とは異なる場合があります")

    except Exception as e:
        logger.error(f"トークン使用量推定エラー: {e}")
        st.error("トークン使用量の推定に失敗しました")


# ==================================================
# データ処理関数群
# ==================================================
def clean_text(text: str) -> str:
    """テキストのクレンジング処理"""
    if pd.isna(text) or text == "":
        return ""

    # 文字列に変換
    text = str(text)

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


def combine_columns(row: pd.Series, dataset_type: str = "customer_support_faq") -> str:
    """複数列を結合して1つのテキストにする"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    # 各列からテキストを抽出・クレンジング
    cleaned_values = []
    for col in required_columns:
        if col in row.index:
            value = row.get(col, '')
            cleaned_text = clean_text(str(value))
            if cleaned_text:  # 空でない場合のみ追加
                cleaned_values.append(cleaned_text)

    # 結合
    combined = " ".join(cleaned_values)
    return combined.strip()


def validate_data(df: pd.DataFrame, dataset_type: str = None) -> List[str]:
    """データの検証"""
    issues = []

    # 基本統計
    issues.append(f"総行数: {len(df):,}")
    issues.append(f"総列数: {len(df.columns)}")

    # 必須列の確認
    if dataset_type:
        config_data = RAGConfig.get_config(dataset_type)
        required_columns = config_data["required_columns"]

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
            issues.append(f"{col}列: 空値 {empty_count:,}個 ({percentage:.1f}%)")

    # 重複行の確認
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"⚠️ 重複行: {duplicate_count:,}個")
    else:
        issues.append("✅ 重複行なし")

    return issues


@safe_execute
def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """データセットの読み込みと基本検証"""
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)

    # 基本検証
    validation_results = validate_data(df, dataset_type)

    logger.info(f"データセット読み込み完了: {len(df):,}行, {len(df.columns)}列")
    return df, validation_results


@safe_execute
def process_rag_data(df: pd.DataFrame, dataset_type: str, combine_columns_option: bool = True) -> pd.DataFrame:
    """RAGデータの前処理を実行"""
    # 基本的な前処理
    df_processed = df.copy()

    # 重複行の除去
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = initial_rows - len(df_processed)

    # 空行の除去（全列がNAの行）
    df_processed = df_processed.dropna(how='all')
    empty_rows_removed = initial_rows - duplicates_removed - len(df_processed)

    # インデックスのリセット
    df_processed = df_processed.reset_index(drop=True)

    logger.info(f"前処理完了: 重複除去={duplicates_removed:,}行, 空行除去={empty_rows_removed:,}行")

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

        # 空の結合テキストを除去
        before_filter = len(df_processed)
        df_processed = df_processed[df_processed['Combined_Text'].str.strip() != '']
        after_filter = len(df_processed)
        empty_combined_removed = before_filter - after_filter

        if empty_combined_removed > 0:
            logger.info(f"空の結合テキストを除去: {empty_combined_removed:,}行")

    return df_processed


@safe_execute
def create_download_data(df: pd.DataFrame, include_combined: bool = True, dataset_type: str = None) -> Tuple[
    str, Optional[str]]:
    """ダウンロード用データの作成"""
    # CSVデータの作成
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_data = csv_buffer.getvalue()

    # 結合テキストデータの作成
    text_data = None
    if include_combined and 'Combined_Text' in df.columns:
        # インデックスなしで結合テキストのみを出力
        text_lines = []
        for text in df['Combined_Text']:
            if text and str(text).strip():
                text_lines.append(str(text).strip())
        text_data = '\n'.join(text_lines)

    return csv_data, text_data


def display_statistics(df_original: pd.DataFrame, df_processed: pd.DataFrame, dataset_type: str = None) -> None:
    """処理前後の統計情報を表示"""
    st.subheader("📊 統計情報")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("元の行数", f"{len(df_original):,}")
    with col2:
        st.metric("処理後の行数", f"{len(df_processed):,}")
    with col3:
        removed_rows = len(df_original) - len(df_processed)
        st.metric("除去された行数", f"{removed_rows:,}")

    # 結合テキストの分析
    if 'Combined_Text' in df_processed.columns:
        st.subheader("📝 結合後テキスト分析")
        text_lengths = df_processed['Combined_Text'].str.len()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均文字数", f"{text_lengths.mean():.0f}")
        with col2:
            st.metric("最大文字数", f"{text_lengths.max():,}")
        with col3:
            st.metric("最小文字数", f"{text_lengths.min():,}")

        # パーセンタイル表示
        percentiles = text_lengths.quantile([0.25, 0.5, 0.75])
        st.write("**文字数分布:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"25%点: {percentiles[0.25]:.0f}文字")
        with col2:
            st.write(f"中央値: {percentiles[0.5]:.0f}文字")
        with col3:
            st.write(f"75%点: {percentiles[0.75]:.0f}文字")


# ==================================================
# ファイル保存関数群
# ==================================================
def create_output_directory() -> Path:
    """OUTPUTディレクトリの作成"""
    try:
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)

        # 書き込み権限のテスト
        test_file = output_dir / ".test_write"
        try:
            test_file.write_text("test", encoding='utf-8')
            if test_file.exists():
                test_file.unlink()
                logger.info("書き込み権限テスト: 成功")
        except Exception as e:
            raise PermissionError(f"書き込み権限テストに失敗: {e}")

        logger.info(f"OUTPUTディレクトリ準備完了: {output_dir.absolute()}")
        return output_dir

    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        raise


@safe_execute
def save_files_to_output(df_processed, dataset_type: str, csv_data: str, text_data: str = None) -> Dict[str, str]:
    """処理済みデータをOUTPUTフォルダに保存"""
    output_dir = create_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # CSVファイルの保存
    csv_filename = f"preprocessed_{dataset_type}_{len(df_processed)}rows_{timestamp}.csv"
    csv_path = output_dir / csv_filename

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_data)

    if csv_path.exists():
        saved_files['csv'] = str(csv_path)
        logger.info(f"CSVファイル保存完了: {csv_path}")

    # テキストファイルの保存
    if text_data and len(text_data.strip()) > 0:
        txt_filename = f"{dataset_type}.txt"
        txt_path = output_dir / txt_filename

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_data)

        if txt_path.exists():
            saved_files['txt'] = str(txt_path)
            logger.info(f"テキストファイル保存完了: {txt_path}")

    # メタデータファイルの保存
    metadata = {
        "dataset_type"        : dataset_type,
        "processed_rows"      : len(df_processed),
        "processing_timestamp": timestamp,
        "created_at"          : datetime.now().isoformat(),
        "files_created"       : list(saved_files.keys()),
        "processing_info"     : {
            "original_rows": st.session_state.get('original_rows', 0),
            "removed_rows" : st.session_state.get('original_rows', 0) - len(df_processed)
        }
    }

    metadata_filename = f"metadata_{dataset_type}_{timestamp}.json"
    metadata_path = output_dir / metadata_filename

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if metadata_path.exists():
        saved_files['metadata'] = str(metadata_path)
        logger.info(f"メタデータファイル保存完了: {metadata_path}")

    return saved_files


# ==================================================
# カスタマーサポートFAQ特有の処理関数
# ==================================================
def validate_customer_support_data_specific(df) -> List[str]:
    """カスタマーサポートFAQデータ特有の検証"""
    support_issues = []

    # サポート関連用語の存在確認
    support_keywords = [
        '問題', '解決', 'トラブル', 'エラー', 'サポート', 'ヘルプ', '対応',
        'problem', 'issue', 'error', 'help', 'support', 'solution', 'troubleshoot'
    ]

    if 'question' in df.columns:
        questions_with_support_terms = 0
        for _, row in df.iterrows():
            question_text = str(row.get('question', '')).lower()
            if any(keyword in question_text for keyword in support_keywords):
                questions_with_support_terms += 1

        support_ratio = (questions_with_support_terms / len(df)) * 100
        support_issues.append(f"サポート関連用語を含む質問: {questions_with_support_terms:,}件 ({support_ratio:.1f}%)")

    # 回答の長さ分析
    if 'answer' in df.columns:
        answer_lengths = df['answer'].astype(str).str.len()
        avg_answer_length = answer_lengths.mean()
        if avg_answer_length < 50:
            support_issues.append(f"⚠️ 平均回答長が短い可能性: {avg_answer_length:.0f}文字")
        else:
            support_issues.append(f"✅ 適切な回答長: 平均{avg_answer_length:.0f}文字")

    # 質問の種類分析（簡易版）
    if 'question' in df.columns:
        question_starters = ['どうすれば', 'なぜ', 'いつ', 'どこで', 'どのように',
                             'what', 'how', 'why', 'when', 'where']
        question_type_count = 0
        for _, row in df.iterrows():
            question_text = str(row.get('question', '')).lower()
            if any(starter in question_text for starter in question_starters):
                question_type_count += 1

        question_type_ratio = (question_type_count / len(df)) * 100
        support_issues.append(f"疑問形質問: {question_type_count:,}件 ({question_type_ratio:.1f}%)")

    return support_issues


def show_usage_instructions(dataset_type: str = "customer_support_faq") -> None:
    """使用方法の説明を表示"""
    st.markdown("---")
    st.subheader("📖 使用方法")
    st.markdown(f"""
    ### 📋 前処理手順
    1. **モデル選択**: サイドバーでRAG用途に適したモデルを選択
    2. **CSVファイルアップロード**: question, answer 列を含むCSVファイルを選択
    3. **前処理実行**: 以下の処理が自動で実行されます：
       - 改行・空白の正規化
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    4. **列結合**: Vector Store/RAG用に最適化された自然な文章として結合
    5. **トークン使用量確認**: 選択モデルでのトークン数とコストを推定
    6. **ダウンロード**: 前処理済みデータを各種形式でダウンロード

    ### 🎯 RAG最適化の特徴
    - **自然な文章結合**: ラベルなしで読みやすい文章として結合
    - **OpenAI embedding対応**: text-embedding-ada-002等に最適化
    - **検索性能向上**: 意味的検索の精度向上

    ### 💡 推奨モデル
    - **コスト重視**: gpt-4o-mini, gpt-4.1-mini
    - **品質重視**: gpt-4o, gpt-4.1
    - **推論タスク**: o1-mini, o3-mini（RAG用途には過剰）
    """)


# ==================================================
# メイン処理関数
# ==================================================
def main():
    """メイン処理関数"""

    # データセットタイプの設定
    DATASET_TYPE = "customer_support_faq"

    # ページ設定
    try:
        st.set_page_config(
            page_title="カスタマーサポートFAQ前処理（完全独立版）",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass

    # ヘッダー
    st.title("💬 カスタマーサポートFAQデータ前処理アプリ")
    st.caption("RAG（Retrieval-Augmented Generation）用データ前処理 - 完全独立版")
    st.markdown("---")

    # =================================================
    # サイドバー: モデル選択機能
    # =================================================
    st.sidebar.title("💬 カスタマーサポートFAQ")
    st.sidebar.markdown("---")

    # モデル選択
    selected_model = select_model(key="rag_model_selection")

    # 選択されたモデル情報を表示
    show_model_info(selected_model)

    st.sidebar.markdown("---")

    # 前処理設定
    st.sidebar.header("⚙️ 前処理設定")
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

    # カスタマーサポートデータ特有の設定
    with st.sidebar.expander("💬 サポートデータ設定", expanded=False):
        preserve_formatting = st.checkbox(
            "書式を保護",
            value=True,
            help="回答内の重要な書式を保護"
        )
        normalize_questions = st.checkbox(
            "質問を正規化",
            value=True,
            help="質問文の表記ゆれを統一"
        )

    # =================================================
    # メインエリア: ファイル処理
    # =================================================

    # 現在の選択モデル情報表示
    with st.expander("📊 選択中のモデル情報", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🤖 選択モデル: **{selected_model}**")
        with col2:
            limits = AppConfig.get_model_limits(selected_model)
            st.info(f"📏 最大トークン: **{limits['max_tokens']:,}**")

    # ファイルアップロード
    st.subheader("📁 データファイルのアップロード")
    uploaded_file = st.file_uploader(
        "カスタマーサポートFAQデータのCSVファイルをアップロードしてください",
        type=['csv'],
        help="question, answer の2列を含むCSVファイル"
    )

    if uploaded_file is not None:
        try:
            # ファイル情報の確認
            st.info(f"📁 ファイル: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")

            # セッション状態でファイル処理状況を管理
            file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"

            # ファイルが変更された場合は再読み込み
            if st.session_state.get('current_file_key') != file_key:
                with st.spinner("📖 ファイルを読み込み中..."):
                    df, validation_results = load_dataset(uploaded_file, DATASET_TYPE)

                # セッション状態に保存
                st.session_state['current_file_key'] = file_key
                st.session_state['original_df'] = df
                st.session_state['validation_results'] = validation_results
                st.session_state['original_rows'] = len(df)
                st.session_state['file_processed'] = False

                logger.info(f"新しいファイルを読み込み: {len(df):,}行")
            else:
                # セッション状態から取得
                df = st.session_state['original_df']
                validation_results = st.session_state['validation_results']
                logger.info(f"セッション状態からファイルを取得: {len(df):,}行")

            st.success(f"✅ ファイルが正常に読み込まれました。行数: **{len(df):,}**")

            # 元データの表示
            st.subheader("📋 元データプレビュー")
            st.dataframe(df.head(10), use_container_width=True)

            # データ検証結果の表示
            if show_validation:
                st.subheader("🔍 データ検証")

                # 基本検証結果
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**基本統計:**")
                    for issue in validation_results:
                        st.info(issue)

                with col2:
                    # カスタマーサポートデータ特有の検証
                    support_issues = validate_customer_support_data_specific(df)
                    if support_issues:
                        st.write("**サポートデータ特有の分析:**")
                        for issue in support_issues:
                            st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("前処理を実行すると、データのクレンジング・正規化・結合が行われます。")
            with col2:
                process_button = st.button("🚀 前処理を実行", type="primary", key="process_button",
                                           use_container_width=True)

            if process_button:
                try:
                    with st.spinner("⚙️ 前処理中..."):
                        # RAGデータの前処理
                        df_processed = process_rag_data(
                            df.copy(),
                            DATASET_TYPE,
                            combine_columns_option
                        )

                    st.success("✅ 前処理が完了しました！")

                    # セッション状態に処理済みデータを保存
                    st.session_state['processed_df'] = df_processed
                    st.session_state['file_processed'] = True

                    # 前処理後のデータ表示
                    st.subheader("✅ 前処理後のデータプレビュー")
                    st.dataframe(df_processed.head(10), use_container_width=True)

                    # 統計情報の表示
                    display_statistics(df, df_processed, DATASET_TYPE)

                    # 選択されたモデルでのトークン使用量推定
                    estimate_token_usage(df_processed, selected_model)

                    # カスタマーサポートデータ特有の後処理分析
                    if 'Combined_Text' in df_processed.columns:
                        st.subheader("💬 カスタマーサポートデータ特有の分析")

                        col1, col2 = st.columns(2)

                        with col1:
                            # 結合テキストのサポート用語分析
                            combined_texts = df_processed['Combined_Text']
                            support_keywords = ['問題', 'エラー', 'トラブル', 'サポート', 'ヘルプ']

                            keyword_counts = {}
                            for keyword in support_keywords:
                                count = combined_texts.str.contains(keyword, case=False, na=False).sum()
                                keyword_counts[keyword] = count

                            if keyword_counts:
                                st.write("**サポート関連用語の出現頻度:**")
                                for keyword, count in keyword_counts.items():
                                    percentage = (count / len(df_processed)) * 100
                                    st.write(f"- {keyword}: {count:,}件 ({percentage:.1f}%)")

                        with col2:
                            # 質問の長さ分布
                            if 'question' in df_processed.columns:
                                question_lengths = df_processed['question'].str.len()
                                st.write("**質問の長さ統計:**")
                                st.metric("平均質問長", f"{question_lengths.mean():.0f}文字")
                                st.metric("最長質問", f"{question_lengths.max():,}文字")
                                st.metric("最短質問", f"{question_lengths.min():,}文字")

                    logger.info(f"カスタマーサポートFAQデータ処理完了: {len(df):,} → {len(df_processed):,}行")

                except Exception as process_error:
                    st.error(f"❌ 前処理エラー: {str(process_error)}")
                    logger.error(f"前処理エラー: {process_error}")
                    with st.expander("🔧 詳細エラー情報", expanded=False):
                        st.code(str(process_error))

            # 処理済みデータがある場合のみダウンロード・保存セクションを表示
            if st.session_state.get('file_processed', False) and 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']

                # ダウンロード・保存セクション
                st.subheader("💾 ダウンロード・保存")

                # ダウンロード用データの作成（キャッシュ）
                if 'download_data' not in st.session_state or st.session_state.get('download_data_key') != file_key:
                    with st.spinner("📦 ダウンロード用データを準備中..."):
                        csv_data, text_data = create_download_data(
                            df_processed,
                            combine_columns_option,
                            DATASET_TYPE
                        )
                        st.session_state['download_data'] = (csv_data, text_data)
                        st.session_state['download_data_key'] = file_key
                else:
                    csv_data, text_data = st.session_state['download_data']

                # ブラウザダウンロード
                st.write("**📥 ブラウザダウンロード**")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📊 CSV形式でダウンロード",
                        data=csv_data,
                        file_name=f"preprocessed_{DATASET_TYPE}_{len(df_processed)}rows.csv",
                        mime="text/csv",
                        help="前処理済みのカスタマーサポートFAQデータをCSV形式でダウンロード",
                        use_container_width=True
                    )

                with col2:
                    if text_data:
                        st.download_button(
                            label="📝 テキスト形式でダウンロード",
                            data=text_data,
                            file_name=f"customer_support_faq.txt",
                            mime="text/plain",
                            help="Vector Store/RAG用に最適化された結合テキスト",
                            use_container_width=True
                        )
                    else:
                        st.info("結合テキストが利用できません")

                # ローカル保存
                st.write("**💾 ローカルファイル保存（OUTPUTフォルダ）**")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("処理済みデータを OUTPUTフォルダに保存します。")
                with col2:
                    save_button = st.button("🔄 OUTPUTフォルダに保存", type="secondary", key="save_button",
                                            use_container_width=True)

                if save_button:
                    try:
                        with st.spinner("💾 ファイル保存中..."):
                            saved_files = save_files_to_output(
                                df_processed,
                                DATASET_TYPE,
                                csv_data,
                                text_data
                            )

                        if saved_files:
                            st.success("✅ ファイル保存完了！")

                            # 保存されたファイル一覧を表示
                            with st.expander("📂 保存されたファイル一覧", expanded=True):
                                for file_type, file_path in saved_files.items():
                                    if Path(file_path).exists():
                                        file_size = Path(file_path).stat().st_size
                                        st.success(f"**{file_type.upper()}**: `{file_path}` ({file_size:,} bytes)")
                                    else:
                                        st.error(f"**{file_type.upper()}**: `{file_path}` ❌ ファイルが見つかりません")

                                # OUTPUTフォルダの場所を表示
                                output_path = Path("OUTPUT").resolve()
                                st.info(f"**保存場所**: `{output_path}`")
                                try:
                                    file_count = len(list(output_path.glob("*")))
                                    st.info(f"**フォルダ内ファイル数**: {file_count:,}個")
                                except:
                                    st.info("フォルダ情報取得中...")
                        else:
                            st.error("❌ ファイル保存に失敗しました")

                    except Exception as save_error:
                        st.error(f"❌ ファイル保存エラー: {str(save_error)}")
                        logger.error(f"保存エラー: {save_error}")

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            logger.error(f"ファイル読み込みエラー: {e}")

            with st.expander("🔧 詳細エラー情報", expanded=False):
                st.code(str(e))

                # ファイル情報の詳細確認
                if uploaded_file is not None:
                    st.write("**ファイル診断:**")
                    st.write(f"- ファイル名: {uploaded_file.name}")
                    st.write(f"- ファイルサイズ: {uploaded_file.size:,} bytes")
                    st.write(f"- ファイルタイプ: {uploaded_file.type}")

    else:
        st.info("👆 CSVファイルをアップロードしてください")

        # サンプルファイルの説明
        with st.expander("📄 必要なファイル形式", expanded=False):
            st.write("**CSVファイルの要件:**")
            st.write("- エンコーディング: UTF-8")
            st.write("- 必須列: question, answer")
            st.write("- ファイル形式: .csv")

            st.write("**サンプルデータ例:**")
            sample_data = {
                "question": [
                    "パスワードを忘れました",
                    "支払い方法を変更したい",
                    "サービスが利用できません"
                ],
                "answer"  : [
                    "パスワードリセットページからリセットできます",
                    "アカウント設定から支払い方法を変更してください",
                    "システムの状況を確認し、サポートにお問い合わせください"
                ]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)

    # 使用方法の説明
    show_usage_instructions(DATASET_TYPE)

    # セッション状態の表示（デバッグ用）
    if st.sidebar.checkbox("🔧 セッション状態を表示", value=False):
        with st.sidebar.expander("デバッグ情報", expanded=False):
            st.write(f"**選択モデル**: {selected_model}")
            st.write(f"**ファイル処理済み**: {st.session_state.get('file_processed', False)}")

            if 'original_df' in st.session_state:
                df = st.session_state['original_df']
                st.write(f"**元データ**: {len(df):,}行")

            if 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']
                st.write(f"**処理済みデータ**: {len(df_processed):,}行")

            # OUTPUTフォルダの状態
            try:
                output_dir = Path("OUTPUT")
                if output_dir.exists():
                    file_count = len(list(output_dir.glob(f"*{DATASET_TYPE}*")))
                    st.write(f"**保存済みファイル**: {file_count:,}個")
                else:
                    st.write("**OUTPUTフォルダ**: 未作成")
            except Exception as e:
                st.write(f"**フォルダ状態**: エラー ({e})")


# ==================================================
# アプリケーション実行
# ==================================================
if __name__ == "__main__":
    main()

# 実行コマンド:
# streamlit run a30_012_make_rag_data_customer_standalone.py --server.port=8501
