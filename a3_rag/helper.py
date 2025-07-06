# helper_old.py for RAG
# helper_old.py
# RAGシステム用の共通ヘルパー関数

import time
from typing import List, Optional, Dict, Any
from openai import OpenAI
from openai.types.responses import Response


# ==================================================
# Vector Store 関連
# ==================================================
def create_vector_store_and_upload(text_content: str, upload_name: str) -> str:
    """
    テキストコンテンツからVector Storeを作成し、IDを返す

    Args:
        text_content: アップロードするテキスト内容
        upload_name: Vector Storeの名前

    Returns:
        作成されたVector StoreのID
    """
    client = OpenAI()

    # Vector Storeの作成
    vs = client.vector_stores.create(name=upload_name)
    vs_id = vs.id

    # テキストを一時ファイルとしてアップロード
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(text_content)
        tmp_path = tmp.name

    # ファイルをアップロード
    with open(tmp_path, 'rb') as f:
        file_obj = client.files.create(file=f, purpose="assistants")

    # Vector Storeにファイルを追加
    client.vector_stores.files.create(
        vector_store_id=vs_id,
        file_id=file_obj.id
    )

    # インデックス完了を待機
    print(f"Waiting for vector store {vs_id} to be ready...")
    while client.vector_stores.retrieve(vs_id).status != "completed":
        time.sleep(2)

    # 一時ファイルを削除
    import os
    os.unlink(tmp_path)

    print(f"Vector Store ready: {vs_id}")
    return vs_id


def standalone_search(query: str, vs_id: str, max_results: int = 5) -> str:
    """
    Vector Storeで検索を実行し、結果をテキストで返す

    Args:
        query: 検索クエリ
        vs_id: Vector Store ID
        max_results: 最大結果数

    Returns:
        検索結果のテキスト
    """
    client = OpenAI()

    try:
        results = client.vector_stores.search(
            vector_store_id=vs_id,
            query=query,
            max_results=max_results
        )

        # 結果をテキストに整形
        output_texts = []
        for i, result in enumerate(results.data, 1):
            score = result.score
            content = result.content[0].text.strip() if result.content else ""
            output_texts.append(f"[Result {i} - Score: {score:.3f}]\n{content}")

        return "\n\n".join(output_texts) if output_texts else "No results found."

    except Exception as e:
        return f"Search error: {str(e)}"


# ==================================================
# Response 処理
# ==================================================
def extract_text_from_response(response: Response) -> List[str]:
    """
    Responseオブジェクトからテキストを抽出

    Args:
        response: OpenAI Response オブジェクト

    Returns:
        抽出されたテキストのリスト
    """
    texts = []

    if hasattr(response, 'output'):
        for item in response.output:
            if hasattr(item, 'type') and item.type == "message":
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'type') and content.type == "output_text":
                            if hasattr(content, 'text'):
                                texts.append(content.text)

    # フォールバック: output_text属性
    if not texts and hasattr(response, 'output_text'):
        texts.append(response.output_text)

    return texts


# ==================================================
# Vector Store 検索の高度な機能
# ==================================================
def search_with_metadata(vs_id: str, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    メタデータフィルタ付きでVector Store検索を実行

    Args:
        vs_id: Vector Store ID
        query: 検索クエリ
        filters: メタデータフィルタ

    Returns:
        検索結果のリスト
    """
    client = OpenAI()

    search_params = {
        "vector_store_id": vs_id,
        "query"          : query,
        "max_results"    : 10
    }

    if filters:
        search_params["metadata_filters"] = filters

    try:
        results = client.vector_stores.search(**search_params)

        return [
            {
                "score"   : r.score,
                "content" : r.content[0].text.strip() if r.content else "",
                "metadata": r.metadata if hasattr(r, 'metadata') else {}
            }
            for r in results.data
        ]
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []


# ==================================================
# バッチ処理用ヘルパー
# ==================================================
def batch_create_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    テキストのリストをバッチで埋め込みベクトルに変換

    Args:
        texts: テキストのリスト
        batch_size: バッチサイズ

    Returns:
        埋め込みベクトルのリスト
    """
    client = OpenAI()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([data.embedding for data in response.data])

    return embeddings


# ==================================================
# デバッグ用ヘルパー
# ==================================================
def inspect_vector_store(vs_id: str) -> Dict[str, Any]:
    """
    Vector Storeの情報を取得

    Args:
        vs_id: Vector Store ID

    Returns:
        Vector Storeの詳細情報
    """
    client = OpenAI()

    try:
        vs = client.vector_stores.retrieve(vs_id)

        # ファイル一覧を取得
        files = client.vector_stores.files.list(vector_store_id=vs_id)

        return {
            "id"        : vs.id,
            "name"      : vs.name,
            "status"    : vs.status,
            "file_count": len(files.data) if files else 0,
            "created_at": vs.created_at,
            "metadata"  : vs.metadata if hasattr(vs, 'metadata') else {}
        }
    except Exception as e:
        return {"error": str(e)}


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    'create_vector_store_and_upload',
    'standalone_search',
    'extract_text_from_response',
    'search_with_metadata',
    'batch_create_embeddings',
    'inspect_vector_store',
]
