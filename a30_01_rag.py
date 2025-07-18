# a30_01_rag.py - OpenAI Agent SDK版
# a30_01_rag.py - OpenAI Agent SDK/Responses API版
import streamlit as st
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import traceback

# OpenAI Agent SDK のインポート
try:
    from openai import OpenAI
    from openai_agents import Agent, Runner, Session, FileSearchTool
    from openai_agents.tools import function_tool
    from openai_agents.streaming import StreamingRunner

    AGENT_SDK_AVAILABLE = True
except ImportError:
    # フォールバック: 従来のOpenAI API
    from openai import OpenAI

    AGENT_SDK_AVAILABLE = False
    st.warning(
        "OpenAI Agent SDK が見つかりません。従来のAPI使用します。`pip install openai-agents` でインストールしてください。")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector Store IDの設定
VECTOR_STORES = {
    "Customer Support FAQ"    : "vs_687a0604f1508191aaf416d88e266ab7",
    "Science & Technology Q&A": "vs_687a061acc908191af7d5d9ba623470b",
    "Medical Q&A"             : "vs_687a060f9ed881918b213bfdeab8241b",
    "Legal Q&A"               : "vs_687a062418ec8191872efdbf8f554836"
}

# OpenAI APIキーの設定
try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"OpenAI API キーの設定に問題があります: {e}")
    st.stop()


class RAGAgentManager:
    """RAGエージェントの管理クラス"""

    def __init__(self):
        self.agents = {}
        self.sessions = {}
        self.initialize_agents()

    def initialize_agents(self):
        """各Vector Store用のエージェントを初期化"""
        for store_name, store_id in VECTOR_STORES.items():
            if AGENT_SDK_AVAILABLE:
                self.agents[store_name] = self.create_agent_sdk_rag(store_name, store_id)
            else:
                self.agents[store_name] = self.create_fallback_rag(store_name, store_id)

    def create_agent_sdk_rag(self, store_name: str, store_id: str) -> Agent:
        """Agent SDKを使用したRAGエージェントの作成"""
        instructions = f"""
        あなたは{store_name}の専門検索アシスタントです。
        以下の役割を果たしてください：

        1. 質問に対して関連する情報を検索ツールを使って調べる
        2. 検索結果を基に、正確で有用な回答を日本語で提供する
        3. 情報源がある場合は適切に引用する
        4. 検索結果がない場合は、その旨を明確に伝える
        5. 曖昧な質問には明確化を求める

        常に親切で専門的な対応を心がけ、ユーザーの質問に最適な答えを提供してください。
        """

        agent = Agent(
            name=f"RAG_{store_name.replace(' ', '_')}",
            instructions=instructions,
            tools=[FileSearchTool(vector_store_ids=[store_id])],
            model="gpt-4o-mini"
        )
        return agent

    def create_fallback_rag(self, store_name: str, store_id: str) -> Dict:
        """フォールバック用の従来型RAG設定"""
        return {
            "name"        : store_name,
            "store_id"    : store_id,
            "instructions": f"あなたは{store_name}の専門検索アシスタントです。"
        }

    def get_or_create_session(self, store_name: str, user_id: str = "default") -> Optional[Session]:
        """セッションの取得または作成"""
        if not AGENT_SDK_AVAILABLE:
            return None

        session_key = f"{store_name}_{user_id}"
        if session_key not in self.sessions:
            self.sessions[session_key] = Session(session_key)
        return self.sessions[session_key]

    def search_with_agent_sdk(self, query: str, store_name: str) -> Tuple[str, Dict]:
        """Agent SDKを使用した検索"""
        try:
            agent = self.agents[store_name]
            session = self.get_or_create_session(store_name)

            # 検索実行
            result = Runner.run_sync(agent, query, session=session)

            response_text = result.final_output if hasattr(result, 'final_output') else str(result)

            # メタデータの収集
            metadata = {
                "store_name": store_name,
                "query"     : query,
                "timestamp" : datetime.now().isoformat(),
                "model"     : "gpt-4o-mini",
                "method"    : "agent_sdk"
            }

            # 実行統計があれば追加
            if hasattr(result, 'usage'):
                metadata["usage"] = result.usage

            return response_text, metadata

        except Exception as e:
            error_msg = f"Agent SDK検索でエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg, {"error": str(e), "method": "agent_sdk"}

    def search_with_fallback(self, query: str, store_name: str) -> Tuple[str, Dict]:
        """Responses APIを使ったフォールバック検索"""
        try:
            agent_config = self.agents[store_name]
            store_id = agent_config["store_id"]
            # 最新Responses API でシンプルにRAG
            resp = openai_client.responses.create(
                model="gpt-4o-mini",
                tools=[FileSearchTool(vector_store_ids=[store_id])],
                # tools=[{"type": "file_search", "vector_store_ids": [store_id]}],
                input=query
            )
            # output_text の取得。応答内容が output_text に格納される
            response_text = resp.output_text

            metadata = {
                "store_name": store_name,
                "query"     : query,
                "timestamp" : datetime.now().isoformat(),
                "model"     : "gpt-4o-mini",
                "method"    : "responses_api"
            }

            return response_text, metadata

        except Exception as e:
            error_msg = f"検索でエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg, {"error": str(e), "method": "responses_api"}

    def search(self, query: str, store_name: str) -> Tuple[str, Dict]:
        """統合検索メソッド"""
        if AGENT_SDK_AVAILABLE:
            return self.search_with_agent_sdk(query, store_name)
        else:
            return self.search_with_fallback(query, store_name)

    def stream_search(self, query: str, store_name: str):
        """ストリーミング検索（Agent SDK利用時のみ）"""
        if not AGENT_SDK_AVAILABLE:
            # フォールバックとして通常検索
            result, metadata = self.search(query, store_name)
            yield result, metadata
            return

        try:
            agent = self.agents[store_name]
            session = self.get_or_create_session(store_name)

            # ストリーミング実行
            streaming_runner = StreamingRunner()

            for chunk in streaming_runner.run_stream(agent, query, session=session):
                if hasattr(chunk, 'content'):
                    yield chunk.content, {"streaming": True}
                elif hasattr(chunk, 'delta'):
                    yield chunk.delta, {"streaming": True}
                else:
                    yield str(chunk), {"streaming": True}

        except Exception as e:
            error_msg = f"ストリーミング検索でエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            yield error_msg, {"error": str(e), "method": "stream"}


# グローバルインスタンス
@st.cache_resource
def get_rag_manager():
    """RAGエージェントマネージャーのシングルトン取得"""
    return RAGAgentManager()


def initialize_session_state():
    """セッション状態の初期化"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'selected_store' not in st.session_state:
        st.session_state.selected_store = list(VECTOR_STORES.keys())[0]
    if 'streaming_enabled' not in st.session_state:
        st.session_state.streaming_enabled = AGENT_SDK_AVAILABLE


def display_search_history():
    """検索履歴の表示"""
    st.header("🕒 検索履歴")

    if not st.session_state.search_history:
        st.info("検索履歴がありません")
        return

    # 履歴をタブで表示
    for i, item in enumerate(st.session_state.search_history[:10]):  # 最新10件
        with st.expander(f"履歴 {i + 1}: {item['query'][:50]}..."):
            st.markdown(f"**質問:** {item['query']}")
            st.markdown(f"**Vector Store:** {item['store_name']}")
            st.markdown(f"**実行時間:** {item['timestamp']}")
            st.markdown(f"**検索方法:** {item.get('method', 'unknown')}")

            if st.button("再実行", key=f"rerun_{i}"):
                st.session_state.current_query = item['query']
                st.session_state.selected_store = item['store_name']
                st.rerun()


def display_test_questions():
    """テスト用質問の表示"""
    st.header("💡 テスト用質問")

    test_questions = {
        "Customer Support FAQ"    : [
            "新規アカウントを作るにはどうすればよいですか？",
            "どのような決済方法が利用できますか？",
            "商品を返品することはできますか？",
            "パスワードを忘れてしまいました",
            "サポートチームに連絡する方法を教えてください"
        ],
        "Science & Technology Q&A": [
            "人工知能の最新動向について教えてください",
            "量子コンピューティングの原理とは？",
            "再生可能エネルギーの種類と特徴",
            "遺伝子編集技術の現状と課題",
            "宇宙探査の最新技術について"
        ],
        "Medical Q&A"             : [
            "高血圧の予防方法について",
            "糖尿病の症状と治療法",
            "心臓病のリスクファクター",
            "健康的な食事のガイドライン",
            "運動と健康の関係について"
        ],
        "Legal Q&A"               : [
            "契約書の重要な条項について",
            "知的財産権の保護方法",
            "労働法の基本原則",
            "個人情報保護法の概要",
            "消費者保護法の適用範囲"
        ]
    }

    selected_category = st.selectbox(
        "カテゴリを選択",
        list(test_questions.keys()),
        key="test_category"
    )

    questions = test_questions[selected_category]

    for i, question in enumerate(questions):
        if st.button(f"質問 {i + 1}: {question}", key=f"test_q_{selected_category}_{i}"):
            st.session_state.current_query = question
            st.session_state.selected_store = selected_category
            st.rerun()


def display_system_info():
    """システム情報の表示"""
    with st.expander("🔧 システム情報", expanded=False):
        st.write("**利用可能な機能:**")
        st.write(f"- OpenAI Agent SDK: {'✅' if AGENT_SDK_AVAILABLE else '❌'}")
        st.write(f"- ストリーミング検索: {'✅' if AGENT_SDK_AVAILABLE else '❌'}")
        st.write(f"- セッション管理: {'✅' if AGENT_SDK_AVAILABLE else '❌'}")

        st.write("**Vector Stores:**")
        for name, store_id in VECTOR_STORES.items():
            st.write(f"- {name}: `{store_id}`")

        if st.session_state.search_history:
            st.write(f"**検索履歴:** {len(st.session_state.search_history)} 件")


def main():
    """メイン関数"""
    st.set_page_config(
        page_title="RAG検索テストアプリ（Agent SDK版）",
        page_icon="🔍",
        layout="wide"
    )

    # セッション状態の初期化
    initialize_session_state()

    # RAGマネージャーの取得
    rag_manager = get_rag_manager()

    # ヘッダー
    st.title("🔍 RAG検索テストアプリケーション（Agent SDK版）")

    if AGENT_SDK_AVAILABLE:
        st.success("✅ OpenAI Agent SDK が利用可能です")
    else:
        st.warning("⚠️ OpenAI Agent SDK が利用できません。従来のAPI を使用します。")

    st.markdown("---")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        # Vector Store選択
        selected_store = st.selectbox(
            "Vector Store を選択",
            options=list(VECTOR_STORES.keys()),
            index=list(VECTOR_STORES.keys()).index(st.session_state.selected_store),
            key="store_selection"
        )
        st.session_state.selected_store = selected_store

        # 選択されたVector Store IDを表示
        st.code(VECTOR_STORES[selected_store])

        # ストリーミング設定
        if AGENT_SDK_AVAILABLE:
            streaming_enabled = st.checkbox(
                "ストリーミング検索を有効にする",
                value=st.session_state.streaming_enabled,
                help="リアルタイムで応答を表示します"
            )
            st.session_state.streaming_enabled = streaming_enabled

        # システム情報
        display_system_info()

        # テスト用質問
        with st.expander("💡 テスト用質問", expanded=True):
            display_test_questions()

    # メインコンテンツ
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("❓ 質問入力")

        # 質問入力フォーム
        with st.form("search_form"):
            query = st.text_area(
                "質問を入力してください",
                value=st.session_state.current_query,
                height=100,
                key="query_input"
            )

            submitted = st.form_submit_button("🔍 検索実行", type="primary")

        if submitted and query.strip():
            st.session_state.current_query = query

            # 検索実行
            with col2:
                st.header("🤖 検索結果")

                if st.session_state.streaming_enabled and AGENT_SDK_AVAILABLE:
                    # ストリーミング検索
                    result_container = st.empty()
                    accumulated_result = ""

                    with st.spinner("検索中..."):
                        for chunk, metadata in rag_manager.stream_search(query, selected_store):
                            if isinstance(chunk, str):
                                accumulated_result += chunk
                                result_container.markdown(accumulated_result)
                            time.sleep(0.1)  # 表示の調整

                    final_result = accumulated_result
                    final_metadata = metadata

                else:
                    # 通常検索
                    with st.spinner("検索中..."):
                        final_result, final_metadata = rag_manager.search(query, selected_store)

                    st.markdown("### 🤖 回答")
                    st.markdown(final_result)

                # メタデータ表示
                st.markdown("---")
                st.markdown("### 📊 検索情報")
                st.markdown(f"**使用したVector Store:** {selected_store}")
                st.markdown(f"**Vector Store ID:** `{VECTOR_STORES[selected_store]}`")
                st.markdown(f"**検索クエリ:** {query}")
                st.markdown(f"**検索方法:** {final_metadata.get('method', 'unknown')}")

                # 検索履歴に追加
                history_item = {
                    "query"         : query,
                    "store_name"    : selected_store,
                    "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "method"        : final_metadata.get('method', 'unknown'),
                    "result_preview": final_result[:200] + "..." if len(final_result) > 200 else final_result
                }

                # 重複チェック
                if not any(item['query'] == query and item['store_name'] == selected_store
                           for item in st.session_state.search_history):
                    st.session_state.search_history.insert(0, history_item)
                    st.session_state.search_history = st.session_state.search_history[:50]  # 最新50件保持

        elif submitted and not query.strip():
            st.error("質問を入力してください")

    with col2:
        if not st.session_state.current_query:
            st.header("🤖 検索結果")
            st.info("質問を入力して検索を実行してください")

    # 検索履歴セクション
    st.markdown("---")
    display_search_history()

    # フッター
    st.markdown("---")
    st.markdown("**RAG検索テストアプリケーション（Agent SDK版）**")
    st.markdown("OpenAI Agent SDK を使用した次世代RAGアプリケーション")


if __name__ == "__main__":
    main()
