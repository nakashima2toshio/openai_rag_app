# a3_05_local_vector_stores.py
import os
import random
import string

from openai import OpenAI

from datasets import load_dataset
import pandas as pd

"""
▫️[for local DB] 全体の流れ、手順
① データセット準備・加工
② ベクターDB（Pinecone）の設定
③ データの埋め込みとベクター登録
④ 質問をベクター検索で処理
⑤ Responses APIを用いて回答生成
⑥ 複数ツールの統合（マルチツール呼び出し）
"""
# --------------------------------------
"""
① データセット準備・加工
https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
・Hugging Faceなどから質問回答データセットを取得。
・「質問」と「回答」を一つの文字列として結合（マージ）する。
"""
def from_huggingface_dataset():
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split='train[:100]')
    ds_dataframe = pd.DataFrame(ds)

    # 質問と回答を一つのテキストに統合
    ds_dataframe['merged'] = ds_dataframe.apply(
        lambda row: f"Question: {row['Question']} Answer: {row['Response']}", axis=1
    )
    return ds_dataframe

"""
② ベクターDB（Pinecone）の設定
・目的：検索・抽出用ベクターDBを初期化し、データを保持できるように設定。

手順：
・Pineconeに接続し、インデックスを作成。
・埋め込み次元は、OpenAI Embeddings APIを用いて事前に確認。
""" # ------------------------------------------------------
def set_vector_db(ds_dataframe, ServerlessSpec=None):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sample_embedding_resp = client.embeddings.create(
        input=[ds_dataframe['merged'].iloc[0]],
        model="text-embedding-3-small"
    )
    embed_dim = len(sample_embedding_resp.data[0].embedding)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    # ランダムなインデックス名を作成
    index_name = 'pinecone-index-' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

    pc.create_index(
        index_name,
        dimension=embed_dim,
        metric='dotproduct',
        spec=spec
    )

    index = pc.Index(index_name)
    return index


"""
③ データの埋め込みとベクター登録
目的：データを埋め込みベクターに変換し、メタデータ（質問と回答）付きでDBに格納。
手順：
・データをバッチ処理で埋め込みベクター化し、Pineconeにアップサート（更新・登録）。
""" # ------------------------------------------------------
def embedding_dataset(ds_dataframe, index):
    batch_size = 32
    MODEL = "text-embedding-3-small"

    for i in range(0, len(ds_dataframe['merged']), batch_size):
        lines_batch = ds_dataframe['merged'][i: i + batch_size]
        ids_batch = [str(n) for n in range(i, i + len(lines_batch))]

        client = OpenAI()
        res = client.embeddings.create(input=list(lines_batch), model=MODEL)
        embeds = [record.embedding for record in res.data]

        meta = [
            {"Question": rec['Question'], "Answer": rec['Response']}
            for rec in ds_dataframe.iloc[i:i + batch_size].to_dict('records')
        ]

        vectors = list(zip(ids_batch, embeds, meta))
        index.upsert(vectors=vectors)


"""
④ 質問をベクター検索で処理
目的：ユーザーの質問に対し、ベクター検索で関連する質問回答ペアを抽出。
手順：
・質問を埋め込みベクター化し、Pineconeから類似データを取得。
""" # ------------------------------------------------------
def query_pinecone_index(client, index, model, query_text):
    query_embedding = client.embeddings.create(input=query_text, model=model).data[0].embedding
    res = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    return res


"""
⑤ Responses APIを用いて回答生成
目的：ベクター検索結果をコンテキストとして、OpenAI Responses APIを使用して最終的な回答を生成。
手順：
・Pineconeの結果から得たコンテキストを用いてResponses APIに質問し、回答を生成。
""" # ------------------------------------------------------
def get_answer(index):
    matches = index.query(vector=[query_embedding], top_k=3, include_metadata=True)['matches']

    context = "\n\n".join(
        f"Question: {m['metadata']['Question']}\nAnswer: {m['metadata']['Answer']}"
        for m in matches
    )
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=f"Context: {context}\nQuestion: {query_text}"
    )
    print(response.output_text)


"""
⑥ 複数ツールの統合（マルチツール呼び出し）
目的：Responses APIのマルチツール機能を活用し、ウェブ検索やPinecone検索を動的に選択し、最適な回答を自動生成。
手順：
・Toolsを定義し、モデルが適切にツールを選択・実行するよう設定。
""" # ------------------------------------------------------
def multi_tools_integrate():
    tools = [
        {"type": "web_search_preview"},
        {
            "type": "function",
            "name": "PineconeSearchDocuments",
            "parameters": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3}
            }
        }
    ]

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": query_text}],
        tools=tools,
        parallel_tool_calls=True
    )


def main():
    # ① データセット準備・加工
    ds_dataframe = from_huggingface_dataset()

    # ② ベクターDB（Pinecone）の設定
    index = set_vector_db(ds_dataframe)

    # ③ データの埋め込みとベクター登録
    embedding_dataset(ds_dataframe, index)

    # ④ 質問をベクター検索で処理
    query_pinecone_index(client, index, model, query_text)

    # ⑤ Responses APIを用いて回答生成
    get_answer(index)

    # ⑥ 複数ツールの統合（マルチツール呼び出し）
    multi_tools_integrate()

if __name__ == '__main__':
    main()
