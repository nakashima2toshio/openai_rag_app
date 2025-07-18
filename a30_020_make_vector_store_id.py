# a30_020_make_vector_store_id.py
# python a30_020_make_vector_store_id.py
# 4つのデータセットをOpenAI Vector Storeに登録する完全版
# OUTPUT/customer_support_faq.txt
# OUTPUT/medical_qa.txt
# OUTPUT/sciq_qa.txt
# OUTPUT/legal_qa.txt

import os
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

import pandas as pd
import tempfile
import textwrap
from pydantic import BaseModel, Field
from tqdm import tqdm

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ヘルパーモジュールをインポート
try:
    from helper_st import (
        UIHelper, MessageManagerUI, ResponseProcessorUI,
        SessionStateManager, error_handler_ui, timer_ui,
        init_page, select_model, InfoPanelManager
    )
    from helper_api import (
        config, logger, TokenManager, OpenAIClient,
        EasyInputMessageParam, ResponseInputTextParam,
        ConfigManager, MessageManager, sanitize_key,
        error_handler, timer
    )
    from helper_rag import (
        RAGConfig, clean_text, combine_columns,
        additional_preprocessing, validate_data
    )
except ImportError as e:
    # streamlitがインポートされていない場合はスキップ
    logger.warning(f"ヘルパーモジュールのインポートに失敗: {e}")

BASE_DIR = Path(__file__).resolve().parent.parent  # Paslib
THIS_DIR = Path(__file__).resolve().parent  # Paslib
OUTPUT_DIR = THIS_DIR / "OUTPUT"

# ==================================================
# データセット設定
# ==================================================
DATASET_CONFIGS = {
    "customer_support_faq": {
        "filename"   : "customer_support_faq.txt",
        "store_name" : "Customer Support FAQ Knowledge Base",
        "description": "カスタマーサポートFAQデータベース",
        "chunk_size" : 1000,
        "overlap"    : 100
    },
    "medical_qa"          : {
        "filename"   : "medical_qa.txt",
        "store_name" : "Medical Q&A Knowledge Base",
        "description": "医療質問回答データベース",
        "chunk_size" : 1500,
        "overlap"    : 150
    },
    "sciq_qa"             : {
        "filename"   : "sciq_qa.txt",
        "store_name" : "Science & Technology Q&A Knowledge Base",
        "description": "科学技術質問回答データベース",
        "chunk_size" : 800,
        "overlap"    : 80
    },
    "legal_qa"            : {
        "filename"   : "legal_qa.txt",
        "store_name" : "Legal Q&A Knowledge Base",
        "description": "法律質問回答データベース",
        "chunk_size" : 1200,
        "overlap"    : 120
    }
}


# ==================================================
# テキストファイル処理関数
# ==================================================
def load_text_file(filepath: Path) -> List[str]:
    """
    テキストファイルを読み込み、行ごとのリストとして返す

    Args:
        filepath: ファイルパス

    Returns:
        List[str]: テキスト行のリスト
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 空行と短すぎる行を除去
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # 10文字以上の行のみ保持
                cleaned_lines.append(line)

        logger.info(f"ファイル読み込み完了: {filepath.name} - {len(cleaned_lines)}行")
        return cleaned_lines

    except FileNotFoundError:
        logger.error(f"ファイルが見つかりません: {filepath}")
        return []
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {filepath} - {e}")
        return []


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    長いテキストを指定サイズのチャンクに分割

    Args:
        text: 分割対象のテキスト
        chunk_size: チャンクサイズ（文字数）
        overlap: オーバーラップサイズ（文字数）

    Returns:
        List[str]: 分割されたテキストチャンク
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 文の境界で分割するように調整
        if end < len(text):
            # 句読点を探す
            for punct in ['。', '！', '？', '.', '!', '?']:
                punct_pos = text.rfind(punct, start, end)
                if punct_pos > start + chunk_size // 2:
                    end = punct_pos + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 次の開始位置を設定（オーバーラップを考慮）
        start = max(start + 1, end - overlap)

        # 無限ループ防止
        if start >= len(text):
            break

    return chunks


def text_to_jsonl_data(lines: List[str], dataset_type: str) -> List[Dict[str, str]]:
    """
    テキスト行をJSONL用のデータ構造に変換

    Args:
        lines: テキスト行のリスト
        dataset_type: データセットタイプ

    Returns:
        List[Dict]: JSONL用データ構造のリスト
    """
    config = DATASET_CONFIGS.get(dataset_type, {})
    chunk_size = config.get('chunk_size', 1000)
    overlap = config.get('overlap', 100)

    jsonl_data = []

    for idx, line in enumerate(lines):
        # テキストクリーニング
        cleaned_text = clean_text(line)

        if not cleaned_text:
            continue

        # 長いテキストをチャンクに分割
        chunks = chunk_text(cleaned_text, chunk_size, overlap)

        for chunk_idx, chunk in enumerate(chunks):
            jsonl_entry = {
                "id"      : f"{dataset_type}_{idx}_{chunk_idx}",
                "text"    : chunk,
                "metadata": {
                    "dataset"      : dataset_type,
                    "original_line": idx,
                    "chunk_index"  : chunk_idx,
                    "total_chunks" : len(chunks)
                }
            }
            jsonl_data.append(jsonl_entry)

    logger.info(f"{dataset_type}: {len(lines)}行 -> {len(jsonl_data)}チャンク")
    return jsonl_data


# ==================================================
# 既存の関数（改良版）
# ==================================================
def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[
    Optional[List[float]]]:
    """
    OpenAI Embedding APIを使用してテキストリストをバッチでベクトル化

    Args:
        texts: ベクトル化するテキストのリスト
        model: 使用するembeddingモデル (推奨: text-embedding-3-small)
        batch_size: 一度のAPI呼び出しで処理するテキスト数（最大2048）

    Returns:
        埋め込みベクトルのリスト
    """
    client = OpenAI()
    embeddings = []

    # テキストを前処理（改行文字をスペースに置換）
    cleaned_texts = [text.replace("\n", " ") for text in texts]

    logger.info(f"Embedding作成開始: {len(cleaned_texts)}件のテキストを{batch_size}件ずつ処理")

    # バッチごとに処理
    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Embedding作成中"):
        batch = cleaned_texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                input=batch,
                model=model,
                # dimensions=1024  # コスト効率を重視する場合は次元数を削減
            )

            # レスポンスからembeddingを取得
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

            logger.info(
                f"バッチ {i // batch_size + 1}/{(len(cleaned_texts) - 1) // batch_size + 1}: {len(batch)}件処理完了")

            # レート制限対応のための短い待機
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"バッチ {i // batch_size + 1}でエラー: {e}")
            # エラーが発生した場合はNoneで埋める
            embeddings.extend([None] * len(batch))

            # 一時的な問題の場合はリトライ
            time.sleep(2)

    return embeddings


def create_vector_store_from_jsonl_data(jsonl_data: List[Dict], store_name: str) -> Optional[str]:
    """
    JSONL形式のデータからVector Storeを作成

    Args:
        jsonl_data: JSONL形式のデータリスト
        store_name: Vector Storeの名前

    Returns:
        Vector Store ID（成功時）またはNone（失敗時）
    """
    client = OpenAI()
    temp_file_path = None
    uploaded_file_id = None

    try:
        # 一時ファイルを作成してアップロード用のJSONLファイルを準備
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            for entry in jsonl_data:
                # メタデータは文字列として保存（OpenAI側の制限対応）
                jsonl_entry = {
                    "id"  : entry["id"],
                    "text": entry["text"]
                }
                temp_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

            temp_file_path = temp_file.name

        logger.info(f"JSONLファイル作成完了: {temp_file_path} ({len(jsonl_data)}エントリ)")

        # Step 1: ファイルをOpenAIにアップロード
        with open(temp_file_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            uploaded_file_id = uploaded_file.id

        logger.info(f"ファイルアップロード完了: File ID={uploaded_file_id}")

        # Step 2: Vector Storeを作成
        vector_store = client.vector_stores.create(
            name=store_name,
            metadata={
                "created_by" : "vector_store_creator",
                "version"    : "2025.1",
                "data_format": "jsonl_as_txt",
                "entry_count": str(len(jsonl_data))
            }
        )

        logger.info(f"Vector Store作成完了: ID={vector_store.id}")

        # Step 3: ファイルをVector Storeに追加
        vector_store_file = client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=uploaded_file_id
        )

        logger.info(f"Vector StoreFileリンク作成: {vector_store_file.id}")

        # Step 4: ファイル処理完了を待機
        max_wait_time = 600  # 最大10分待機
        wait_interval = 5  # 5秒間隔でチェック
        waited_time = 0

        while waited_time < max_wait_time:
            file_status = client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=uploaded_file_id
            )

            logger.info(f"ファイル処理状況: {file_status.status} (待機時間: {waited_time}秒)")

            if file_status.status == "completed":
                updated_vector_store = client.vector_stores.retrieve(vector_store.id)

                logger.info(f"✅ Vector Store作成完了:")
                logger.info(f"  - ID: {vector_store.id}")
                logger.info(f"  - Name: {vector_store.name}")
                logger.info(f"  - ファイル処理状況: {file_status.status}")
                logger.info(f"  - 総ファイル数: {updated_vector_store.file_counts.total}")
                logger.info(f"  - 完了ファイル数: {updated_vector_store.file_counts.completed}")
                logger.info(f"  - 失敗ファイル数: {updated_vector_store.file_counts.failed}")
                logger.info(f"  - ストレージ使用量: {updated_vector_store.usage_bytes} bytes")

                return vector_store.id

            elif file_status.status == "failed":
                logger.error(f"❌ ファイル処理失敗: {file_status.last_error}")
                return None

            elif file_status.status in ["in_progress", "cancelling"]:
                time.sleep(wait_interval)
                waited_time += wait_interval
            else:
                logger.warning(f"⚠️ 予期しないステータス: {file_status.status}")
                time.sleep(wait_interval)
                waited_time += wait_interval

        logger.error(f"❌ Vector Store作成タイムアウト (制限時間: {max_wait_time}秒)")
        return None

    except Exception as e:
        logger.error(f"Vector Store作成エラー: {e}")
        logger.error(f"エラータイプ: {type(e).__name__}")

        # 具体的なエラー対応の提案
        if "authentication" in str(e).lower():
            logger.error("🔑 APIキーを確認してください。環境変数OPENAI_API_KEYが正しく設定されているか確認。")
        elif "quota" in str(e).lower() or "limit" in str(e).lower():
            logger.error("💳 APIクオータまたはレート制限に達しています。料金プランまたは使用量を確認してください。")
        elif "file" in str(e).lower():
            logger.error("📁 ファイル関連のエラーです。ファイルサイズやフォーマットを確認してください。")

        return None

    finally:
        # 一時ファイルを削除
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("🗑️ 一時ファイルを削除しました")


# ==================================================
# Vector Store管理クラス
# ==================================================
class VectorStoreManager:
    """Vector Store管理クラス"""

    def __init__(self):
        self.client = OpenAI()
        self.created_stores = {}

    def process_all_datasets(self, output_dir: Path = OUTPUT_DIR) -> Dict[str, Optional[str]]:
        """
        全データセットを処理してVector Storeを作成

        Args:
            output_dir: 出力ディレクトリ

        Returns:
            Dict[str, Optional[str]]: データセット名 -> Vector Store ID のマッピング
        """
        results = {}

        logger.info("=== Vector Store作成開始 ===")
        logger.info(f"出力ディレクトリ: {output_dir}")

        for dataset_type, config in DATASET_CONFIGS.items():
            logger.info(f"\n--- {dataset_type} 処理開始 ---")

            # ファイルパスの構築
            filepath = output_dir / config["filename"]

            if not filepath.exists():
                logger.error(f"ファイルが見つかりません: {filepath}")
                results[dataset_type] = None
                continue

            try:
                # Step 1: ファイル読み込み
                logger.info(f"Step 1: ファイル読み込み - {filepath.name}")
                text_lines = load_text_file(filepath)

                if not text_lines:
                    logger.error(f"有効なテキストが見つかりません: {filepath}")
                    results[dataset_type] = None
                    continue

                # Step 2: JSONL形式に変換
                logger.info(f"Step 2: JSONL形式変換")
                jsonl_data = text_to_jsonl_data(text_lines, dataset_type)

                if not jsonl_data:
                    logger.error(f"JSONL変換に失敗: {dataset_type}")
                    results[dataset_type] = None
                    continue

                # Step 3: Vector Store作成
                logger.info(f"Step 3: Vector Store作成")
                store_name = config["store_name"]
                vector_store_id = create_vector_store_from_jsonl_data(jsonl_data, store_name)

                if vector_store_id:
                    logger.info(f"✅ {dataset_type} Vector Store作成成功: {vector_store_id}")
                    self.created_stores[dataset_type] = vector_store_id
                    results[dataset_type] = vector_store_id
                else:
                    logger.error(f"❌ {dataset_type} Vector Store作成失敗")
                    results[dataset_type] = None

            except Exception as e:
                logger.error(f"❌ {dataset_type} 処理中にエラー: {e}")
                results[dataset_type] = None

        # 結果サマリー
        self._print_summary(results)
        return results

    def _print_summary(self, results: Dict[str, Optional[str]]):
        """処理結果のサマリーを表示"""
        logger.info("\n=== 処理結果サマリー ===")

        successful = {k: v for k, v in results.items() if v is not None}
        failed = {k: v for k, v in results.items() if v is None}

        logger.info(f"成功: {len(successful)}/{len(results)} データセット")
        logger.info(f"失敗: {len(failed)}/{len(results)} データセット")

        if successful:
            logger.info("\n✅ 成功したVector Store:")
            for dataset, store_id in successful.items():
                store_name = DATASET_CONFIGS[dataset]["store_name"]
                logger.info(f"  - {dataset}: {store_name}")
                logger.info(f"    ID: {store_id}")

        if failed:
            logger.info("\n❌ 失敗したデータセット:")
            for dataset in failed:
                logger.info(f"  - {dataset}: {DATASET_CONFIGS[dataset]['filename']}")

    def save_results(self, results: Dict[str, Optional[str]], output_file: str = "vector_store_ids.json"):
        """結果をJSONファイルに保存"""
        output_path = THIS_DIR / output_file

        # 保存用データの準備
        save_data = {
            "created_at"       : time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_datasets"   : len(DATASET_CONFIGS),
            "successful_stores": len([v for v in results.values() if v is not None]),
            "vector_stores"    : {}
        }

        for dataset, store_id in results.items():
            if store_id:
                save_data["vector_stores"][dataset] = {
                    "vector_store_id": store_id,
                    "store_name"     : DATASET_CONFIGS[dataset]["store_name"],
                    "description"    : DATASET_CONFIGS[dataset]["description"]
                }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"📄 結果をファイルに保存しました: {output_path}")
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")

    def list_vector_stores(self) -> List[Dict]:
        """既存のVector Storeを一覧表示"""
        try:
            stores = self.client.vector_stores.list()
            store_list = []

            for store in stores.data:
                store_info = {
                    "id"         : store.id,
                    "name"       : store.name,
                    "file_counts": store.file_counts.total if store.file_counts else 0,
                    "created_at" : store.created_at,
                    "usage_bytes": store.usage_bytes
                }
                store_list.append(store_info)

            return store_list
        except Exception as e:
            logger.error(f"Vector Store一覧取得エラー: {e}")
            return []


# ==================================================
# メイン実行部分
# ==================================================
def main():
    """メイン実行関数"""
    logger.info("🚀 Vector Store作成プログラム開始")

    # 環境確認
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY環境変数が設定されていません")
        return

    # 出力ディレクトリの確認
    if not OUTPUT_DIR.exists():
        logger.error(f"❌ 出力ディレクトリが見つかりません: {OUTPUT_DIR}")
        logger.info("以下のファイルが必要です:")
        for config in DATASET_CONFIGS.values():
            logger.info(f"  - {OUTPUT_DIR / config['filename']}")
        return

    # 必要ファイルの存在確認
    missing_files = []
    for dataset_type, config in DATASET_CONFIGS.items():
        filepath = OUTPUT_DIR / config["filename"]
        if not filepath.exists():
            missing_files.append(filepath)

    if missing_files:
        logger.error("❌ 以下のファイルが見つかりません:")
        for filepath in missing_files:
            logger.error(f"  - {filepath}")
        return

    # Vector Store Manager を作成して処理実行
    manager = VectorStoreManager()

    try:
        # 全データセットを処理
        results = manager.process_all_datasets()

        # 結果を保存
        manager.save_results(results)

        # 既存のVector Store一覧も表示
        logger.info("\n=== 既存Vector Store一覧 ===")
        existing_stores = manager.list_vector_stores()
        if existing_stores:
            for store in existing_stores[:10]:  # 最新10件
                logger.info(f"  {store['name']}: {store['id']}")
        else:
            logger.info("  Vector Storeが見つかりません")

    except KeyboardInterrupt:
        logger.info("\n⏹️ ユーザーによって処理が中断されました")
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        raise

    logger.info("🏁 Vector Store作成プログラム終了")


# ==================================================
# 個別実行用関数
# ==================================================
def create_single_vector_store(dataset_type: str, output_dir: Path = OUTPUT_DIR) -> Optional[str]:
    """
    単一のデータセットからVector Storeを作成

    Args:
        dataset_type: データセットタイプ
        output_dir: 出力ディレクトリ

    Returns:
        Vector Store ID（成功時）またはNone（失敗時）
    """
    if dataset_type not in DATASET_CONFIGS:
        logger.error(f"未知のデータセットタイプ: {dataset_type}")
        return None

    config = DATASET_CONFIGS[dataset_type]
    filepath = output_dir / config["filename"]

    if not filepath.exists():
        logger.error(f"ファイルが見つかりません: {filepath}")
        return None

    try:
        # ファイル読み込み
        text_lines = load_text_file(filepath)

        # JSONL形式に変換
        jsonl_data = text_to_jsonl_data(text_lines, dataset_type)

        # Vector Store作成
        store_name = config["store_name"]
        vector_store_id = create_vector_store_from_jsonl_data(jsonl_data, store_name)

        return vector_store_id

    except Exception as e:
        logger.error(f"エラー: {e}")
        return None


if __name__ == "__main__":
    main()
