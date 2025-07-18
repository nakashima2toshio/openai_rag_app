# a30_010_make_vector_store_id.py
# OUTPUT/customer_support_faq.txt
# OUTPUT/medical_qa.txt
# OUTPUT/sciq_qa.txt
# OUTPUT/legal_qa.txt

import os
import re
import time
import json
from pathlib import Path
from typing import List, Optional
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

from datasets import load_dataset
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
except ImportError as e:
    # streamlitがインポートされていない場合はスキップ
    pass

BASE_DIR = Path(__file__).resolve().parent.parent  # Paslib
THIS_DIR = Path(__file__).resolve().parent  # Paslib


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


def create_vector_store_from_dataframe(df_clean: pd.DataFrame, store_name: str = "Customer Support FAQ") -> Optional[
    str]:
    """
    DataFrameからVector Storeを作成（最新API対応版）

    Args:
        df_clean: クレンジング済みのDataFrame
        store_name: Vector Storeの名前

    Returns:
        Vector Store ID（成功時）またはNone（失敗時）
    """
    client = OpenAI()
    temp_file_path = None
    uploaded_file_id = None

    try:
        # 一時ファイルを作成してアップロード用のJSONLファイルを準備
        # 拡張子を.txtに変更（OpenAI Files APIの制限対応）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            for idx, row in df_clean.iterrows():
                # JSONL形式でデータを書き込み（拡張子は.txtだが中身はJSONL）
                json_line = {
                    "id"  : f"faq_{idx}",
                    "text": row['combined_text']
                }
                temp_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')

            temp_file_path = temp_file.name

        logger.info(f"JSONLファイル作成完了: {temp_file_path}")

        # Step 1: ファイルをOpenAIにアップロード（.txt拡張子でアップロード）
        with open(temp_file_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            uploaded_file_id = uploaded_file.id

        logger.info(f"ファイルアップロード完了: File ID={uploaded_file_id}")

        # Step 2: Vector Storeを作成（最新API仕様）
        # 型チェックの警告を避けるため、動的に作成
        vector_store = client.vector_stores.create(
            name=store_name,
            file_ids=[uploaded_file_id],  # ファイルIDを直接指定する方法に変更
            metadata={
                "created_by" : "customer_support_faq_processor",
                "version"    : "2025.1",
                "data_format": "jsonl_as_txt"
            }
        )

        logger.info(f"Vector Store作成完了: ID={vector_store.id}")

        # Step 3: ファイル処理完了を待機
        max_wait_time = 300  # 最大5分待機
        wait_interval = 5  # 5秒間隔でチェック
        waited_time = 0

        while waited_time < max_wait_time:
            # ファイルステータスを確認
            file_status = client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=uploaded_file_id
            )

            logger.info(f"ファイル処理状況: {file_status.status} (待機時間: {waited_time}秒)")

            if file_status.status == "completed":
                # Vector Store全体のステータスを確認
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
                # 処理中の場合は継続して待機
                time.sleep(wait_interval)
                waited_time += wait_interval
            else:
                logger.warning(f"⚠️ 予期しないステータス: {file_status.status}")
                time.sleep(wait_interval)
                waited_time += wait_interval

        # タイムアウトの場合
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
        elif "extension" in str(e).lower() or "format" in str(e).lower():
            logger.error("📄 ファイル拡張子またはフォーマットの問題です。サポートされている形式を確認してください。")

        return None

    finally:
        # 一時ファイルを削除
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("🗑️ 一時ファイルを削除しました")


def create_vector_store_with_advanced_options(df_clean: pd.DataFrame,
                                              store_name: str = "Customer Support FAQ") -> Optional[str]:
    """
    高度なオプション付きでVector Storeを作成（型安全版）

    Args:
        df_clean: クレンジング済みのDataFrame
        store_name: Vector Storeの名前

    Returns:
        Vector Store ID（成功時）またはNone（失敗時）
    """
    client = OpenAI()
    temp_file_path = None
    uploaded_file_id = None

    try:
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            for idx, row in df_clean.iterrows():
                json_line = {
                    "id"  : f"faq_{idx}",
                    "text": row['combined_text']
                }
                temp_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            temp_file_path = temp_file.name

        logger.info(f"JSONLファイル作成完了: {temp_file_path}")

        # ファイルアップロード
        with open(temp_file_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            uploaded_file_id = uploaded_file.id

        logger.info(f"ファイルアップロード完了: File ID={uploaded_file_id}")

        # Vector Store作成（シンプル版 - 型エラーを回避）
        vector_store = client.vector_stores.create(
            name=store_name,
            metadata={
                "created_by" : "customer_support_faq_processor",
                "version"    : "2025.1",
                "data_format": "jsonl_as_txt"
            }
        )

        logger.info(f"Vector Store作成完了: ID={vector_store.id}")

        # ファイルをVector Storeに追加
        vector_store_file = client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=uploaded_file_id
        )

        logger.info(f"Vector StoreFileリンク作成: {vector_store_file.id}")

        # 処理完了待機
        max_wait_time = 300
        wait_interval = 5
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
        return None

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("🗑️ 一時ファイルを削除しました")
