# a30_01_rag_sample.py
# -------------------------------------------------------------
# 前処理（Preprocessing）
# -------------------------------------------------------------
# データクリーニング: 欠損値の処理、改行文字の正規化
# テキスト結合: Question、Complex_CoT、Responseを検索しやすい形で結合
# トークン数チェック: OpenAIの8192トークン制限を考慮
# メタデータ準備: 検索結果に必要な情報を保持
# -------------------------------------------------------------
# Embedding処理
# -------------------------------------------------------------
# バッチ処理: API制限を考慮した効率的な処理
# エラーハンドリング: 個別処理へのフォールバック機能
# レート制限対応: 適切な待機時間の設定
# -------------------------------------------------------------
# Vector Store
# -------------------------------------------------------------
# シンプルな実装: ファイルベースの保存・読み込み
# コサイン類似度検索: OpenAI推奨の距離関数
# メタデータ保持: 元の質問や統計情報を維持
# -------------------------------------------------------------
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import json
from typing import List, Dict, Any
import pickle

# Initialize OpenAI client
client = OpenAI()


class MedicalEmbeddingProcessor:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()

    def get_embedding(self, text: str) -> List[float]:
        """
        テキストをembeddingベクトルに変換
        """
        try:
            # 改行文字を空白に置換（OpenAI推奨）
            text = text.replace("\n", " ")

            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        バッチでembeddingを取得（API制限を考慮）
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            try:
                # バッチで処理
                batch_clean = [text.replace("\n", " ") for text in batch]

                response = self.client.embeddings.create(
                    input=batch_clean,
                    model=self.model
                )

                # 結果を取得
                for data in response.data:
                    embeddings.append(data.embedding)

                # API制限を考慮して少し待機
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in batch processing: {e}")
                # エラーが発生した場合は個別に処理
                for text in batch:
                    embedding = self.get_embedding(text)
                    embeddings.append(embedding if embedding else [0.0] * 1536)
                    time.sleep(0.1)

        return embeddings


class SimpleVectorStore:
    # シンプルなベクトルストア実装

    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.texts = []

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict], texts: List[str]):
        # ベクトルとメタデータを追加
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self.texts.extend(texts)
        print(f"Added {len(vectors)} vectors. Total: {len(self.vectors)}")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        # コサイン類似度を計算
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        # クエリベクトルに最も類似するベクトルを検索
        if not self.vectors:
            return []

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self.cosine_similarity(query_vector, vector)
            similarities.append({
                'index'     : i,
                'similarity': similarity,
                'metadata'  : self.metadata[i],
                'text'      : self.texts[i]
            })

        # 類似度でソート
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def save(self, filepath: str):
        # ベクトルストアをファイルに保存
        data = {
            'vectors' : self.vectors,
            'metadata': self.metadata,
            'texts'   : self.texts
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Vector store saved to {filepath}")

    def load(self, filepath: str):
        #
        # ファイルからベクトルストアを読み込み
        #
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.vectors = data['vectors']
        self.metadata = data['metadata']
        self.texts = data['texts']

        print(f"Vector store loaded from {filepath}")


def main():
    # メイン処理：CSV読み込み → Embedding → Vector Store保存
    # 1. 前処理済みデータの読み込み
    print("Loading processed data...")
    df = pd.read_csv("medical_qa_processed.csv")

    # メタデータを辞書形式に変換
    df['metadata'] = df['metadata'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # 2. Embedding処理
    print("Starting embedding process...")
    processor = MedicalEmbeddingProcessor(model="text-embedding-3-small")

    # テキストリストを準備
    texts = df['combined_text'].tolist()
    metadata = df['metadata'].tolist()

    # バッチでembeddingを取得
    embeddings = processor.batch_embeddings(texts, batch_size=50)

    # 3. Vector Storeに保存
    print("Creating vector store...")
    vector_store = SimpleVectorStore()
    vector_store.add_vectors(embeddings, metadata, texts)

    # 4. Vector Storeをファイルに保存
    vector_store.save("medical_qa_vector_store.pkl")

    # 5. 検索テスト
    print("\n=== Search Test ===")
    test_query = "What cardiac condition causes paradoxical embolism?"
    query_embedding = processor.get_embedding(test_query)

    if query_embedding:
        results = vector_store.search(query_embedding, top_k=3)

        print(f"Query: {test_query}")
        print("\nTop 3 similar results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity']:.4f}")
            print(f"Question: {result['metadata']['question'][:100]}...")
            print(f"Index: {result['metadata']['index']}")

    # 6. 統計情報の出力
    print(f"\n=== Final Statistics ===")
    print(f"Total embeddings created: {len(embeddings)}")
    print(f"Vector dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"Average token count: {df['token_count'].mean():.2f}")

    return vector_store


# 検索機能のデモ
def search_demo(vector_store_path: str = "medical_qa_vector_store.pkl"):
    """
    保存されたベクトルストアを使用した検索デモ
    """
    processor = MedicalEmbeddingProcessor()
    vector_store = SimpleVectorStore()
    vector_store.load(vector_store_path)

    while True:
        query = input("\nEnter your medical question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        query_embedding = processor.get_embedding(query)
        if query_embedding:
            results = vector_store.search(query_embedding, top_k=3)

            print(f"\nResults for: '{query}'")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Similarity: {result['similarity']:.4f}")
                # メタデータから元の質問を表示
                original_question = result['metadata']['question']
                print(f"Question: {original_question[:200]}...")


if __name__ == "__main__":
    # メイン処理を実行
    vector_store = main()

    # 検索デモを実行するかユーザーに確認
    demo_choice = input("\nWould you like to run the search demo? (y/n): ")
    if demo_choice.lower() == 'y':
        search_demo()

    # # 1. 前処理
    # processed_df = preprocess_medical_data("medical_qa_clean.csv")
    #
    # # 2. Embedding + Vector Store作成
    # vector_store = main()
    #
    # # 3. 検索
    # search_demo()
