#
import streamlit as st
import openai
from typing import List, Dict
import json

# OpenAI APIキーの設定
openai.api_key = st.secrets["OPENAI_API_KEY"]  # streamlit secrets.tomlまたは環境変数で設定

# Vector Store IDの設定
VECTOR_STORES = {
    "Customer Support FAQ"    : "vs_687a0604f1508191aaf416d88e266ab7",
    "Science & Technology Q&A": "vs_687a061acc908191af7d5d9ba623470b",
    "Medical Q&A"             : "vs_687a060f9ed881918b213bfdeab8241b",
    "Legal Q&A"               : "vs_687a062418ec8191872efdbf8f554836"
}


def search_vector_store(query: str, vector_store_id: str, top_k: int = 5) -> List[Dict]:
    """Vector Storeから関連する文書を検索"""
    try:
        # OpenAI Assistant APIを使用してVector Storeを検索
        client = openai.OpenAI()

        # 一時的なアシスタントを作成
        assistant = client.beta.assistants.create(
            name="RAG Search Assistant",
            instructions="あなたは検索アシスタントです。提供された情報を元に、質問に対して適切な回答を提供してください。",
            model="o4-mini",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )

        # スレッドを作成
        thread = client.beta.threads.create()

        # メッセージを送信
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # 実行
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # 実行完了を待機
        import time
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        # 結果を取得
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # アシスタントを削除
        client.beta.assistants.delete(assistant.id)

        return messages.data[0].content[0].text.value

    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


def main():
    st.set_page_config(
        page_title="RAG検索テストアプリ",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG検索テストアプリケーション")
    st.markdown("OpenAI Vector Storeを使用したRAG検索のテストアプリケーションです。")

    # サイドバーでVector Storeを選択
    with st.sidebar:
        st.header("設定")
        selected_store = st.selectbox(
            "Vector Storeを選択",
            options=list(VECTOR_STORES.keys()),
            index=0
        )

        st.markdown(f"**選択中のVector Store ID:**")
        st.code(VECTOR_STORES[selected_store])

        # テスト用質問の表示
        st.header("テスト用質問（Customer Support FAQ）")
        test_questions = [
            "新規アカウントを作るにはどうすればよいですか？",
            "どのような決済方法が利用できますか？",
            "注文した商品の配送状況を確認したい",
            "商品を返品することはできますか？",
            "注文をキャンセルしたいのですが",
            "商品はどれくらいで届きますか？",
            "海外への配送は可能ですか？",
            "サポートチームに連絡する方法を教えてください",
            "パスワードを忘れてしまいました",
            "登録情報を変更するにはどうすればいいですか？"
        ]

        for i, question in enumerate(test_questions, 1):
            if st.button(f"質問{i}", key=f"test_q_{i}"):
                st.session_state.query = question

    # メインエリア
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("質問入力")
        query = st.text_area(
            "質問を入力してください",
            value=st.session_state.get("query", ""),
            height=100,
            key="query_input"
        )

        if st.button("検索実行", type="primary"):
            if query.strip():
                st.session_state.query = query
                st.session_state.search_executed = True
            else:
                st.error("質問を入力してください")

    with col2:
        st.header("検索結果")

        if st.session_state.get("search_executed", False) and st.session_state.get("query"):
            with st.spinner("検索中..."):
                vector_store_id = VECTOR_STORES[selected_store]
                result = search_vector_store(
                    query=st.session_state.query,
                    vector_store_id=vector_store_id
                )

                st.markdown("### 回答")
                st.markdown(result)

                # 検索情報の表示
                st.markdown("---")
                st.markdown("### 検索情報")
                st.markdown(f"**使用したVector Store:** {selected_store}")
                st.markdown(f"**Vector Store ID:** `{vector_store_id}`")
                st.markdown(f"**検索クエリ:** {st.session_state.query}")
        else:
            st.info("質問を入力して検索を実行してください")

    # 検索履歴の表示
    st.header("検索履歴")
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    if st.session_state.get("search_executed", False):
        # 履歴に追加
        history_item = {
            "query"       : st.session_state.query,
            "vector_store": selected_store,
            "timestamp"   : str(st.session_state.get("timestamp", ""))
        }

        # 重複を避けるため、同じクエリがあれば削除
        st.session_state.search_history = [
            item for item in st.session_state.search_history
            if item["query"] != st.session_state.query
        ]

        st.session_state.search_history.insert(0, history_item)

        # 最新10件のみ保持
        st.session_state.search_history = st.session_state.search_history[:10]

        # search_executedフラグをリセット
        st.session_state.search_executed = False

    # 履歴の表示
    if st.session_state.search_history:
        for i, item in enumerate(st.session_state.search_history):
            with st.expander(f"履歴 {i + 1}: {item['query'][:50]}..."):
                st.markdown(f"**質問:** {item['query']}")
                st.markdown(f"**Vector Store:** {item['vector_store']}")
                if st.button("再実行", key=f"rerun_{i}"):
                    st.session_state.query = item['query']
                    st.rerun()
    else:
        st.info("検索履歴がありません")


if __name__ == "__main__":
    main()
