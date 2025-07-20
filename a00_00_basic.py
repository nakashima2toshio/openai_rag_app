# --------------------------------------------------
# streamlit run a00_00_basic.py --server.port=8501
# [Demoの追加手順]
# (1) デモのクラスを作る
# ・実装すべき関数
#   ・@error_handler
#     def run(self):　実行の手順を書く
#   ・@timer
#     def _process_query(self, model: str, user_input: str)
#       ・class(pydantic)
#       ・messages
#       ・tools
#       ・response = client.responses.create|parse()
# (2) DemoManagerへの登録
# --------------------------------------------------
import os
import sys
import json
import base64
import glob
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseTextConfigParam,
    FileSearchToolParam,
    WebSearchToolParam,
    ComputerToolParam,
)
from openai.types.responses.web_search_tool_param import UserLocation

# プロジェクトディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent

# PYTHONPATHに親ディレクトリを追加
sys.path.insert(0, str(BASE_DIR))

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
    st.error(f"ヘルパーモジュールのインポートに失敗しました: {e}")
    st.stop()

# ページ設定（一度だけ実行）
st.set_page_config(
    page_title=config.get("ui.page_title", "ChatGPT Responses API Demo"),
    page_icon=config.get("ui.page_icon", "🤖"),
    layout=config.get("ui.layout", "wide")
)

# ==================================================
# 基底クラス（改修版）
# ==================================================
class BaseDemo(ABC):
    """デモ機能の基底クラス（情報パネル機能付き）"""

    def __init__(self, demo_name: str):
        self.demo_name = demo_name
        self.config = ConfigManager("config.yml")
        self.client = OpenAI()
        self.safe_key = sanitize_key(demo_name)
        self.message_manager = MessageManagerUI(f"messages_{self.safe_key}")

    def initialize(self):
        """共通の初期化処理"""
        st.write(f"#### {self.demo_name}")

    def select_model(self) -> str:
        """モデル選択UI"""
        return UIHelper.select_model(f"model_{self.safe_key}")

    def setup_sidebar(self, selected_model: str):
        """左サイドバーの情報パネル設定"""
        st.sidebar.write("### 📋 情報パネル")

        # 各情報パネルを表示（モデル情報のみ閉じた状態で開始）
        self._show_model_info_collapsed(selected_model)
        InfoPanelManager.show_session_info()
        InfoPanelManager.show_performance_info()
        InfoPanelManager.show_cost_info(selected_model)
        InfoPanelManager.show_debug_panel()
        InfoPanelManager.show_settings()

    def _show_model_info_collapsed(self, selected_model: str):
        """モデル情報パネル（閉じた状態で開始）"""
        with st.sidebar.expander("📊 モデル情報", expanded=False):
            # 基本情報
            limits = TokenManager.get_model_limits(selected_model)
            pricing = config.get("model_pricing", {}).get(selected_model, {})

            col1, col2 = st.columns(2)
            with col1:
                st.write("最大入力", f"{limits['max_tokens']:,}")
            with col2:
                st.write("最大出力", f"{limits['max_output']:,}")

            # 料金情報
            if pricing:
                st.write("**料金（1000トークンあたり）**")
                st.write(f"- 入力: ${pricing.get('input', 0):.5f}")
                st.write(f"- 出力: ${pricing.get('output', 0):.5f}")

            # モデル特性
            if selected_model.startswith("o"):
                st.info("🧠 推論特化モデル")
            elif "audio" in selected_model:
                st.info("🎵 音声対応モデル")
            elif "gpt-4o" in selected_model:
                st.info("👁️ 視覚対応モデル")

    def handle_error(self, e: Exception):
        """エラーハンドリング"""
        error_msg = config.get("error_messages.network_error", "エラーが発生しました")
        st.error(f"{error_msg}: {str(e)}")
        if st.checkbox("詳細を表示", key=f"error_detail_{self.safe_key}"):
            st.exception(e)

    @abstractmethod
    def run(self):
        """各デモの実行処理"""
        pass


# ==================================================
# テキスト応答デモ（改修版）
# ==================================================
class TextResponseDemo(BaseDemo):
    """基本的なテキスト応答のデモ（情報パネル付き）"""

    @error_handler
    def run(self):
        self.initialize()
        model = self.select_model()
        st.write("選択したモデル:", model)

        st.code("""
messages = self.message_manager.get_default_messages()
messages.append(
    EasyInputMessageParam(role="user", content=user_input)
)

response = self.client.responses.create(
    model=model,
    input=messages
)
        """)

        # 情報パネルの設定
        self.setup_sidebar(model)

        example_query = self.config.get("samples.responses_query",
                                        "OpenAIのAPIで、responses.createを説明しなさい。")
        st.write(f"例: {example_query}")

        with st.form(key=f"text_form_{self.safe_key}"):
            user_input = st.text_area(
                "質問を入力してください:",
                height=self.config.get("ui.text_area_height", 75)
            )
            submitted = st.form_submit_button("送信")

        if submitted and user_input:
            self._process_query(model, user_input)

    @timer
    def _process_query(self, model: str, user_input: str):
        """クエリの処理"""
        try:
            # トークン情報の表示
            UIHelper.show_token_info(user_input, model, position="sidebar")

            messages = self.message_manager.get_default_messages()
            messages.append(
                EasyInputMessageParam(role="user", content=user_input)
            )

            with st.spinner("処理中..."):
                response = self.client.responses.create(
                    model=model,
                    input=messages
                )

            st.success("応答を取得しました")
            ResponseProcessorUI.display_response(response)

        except Exception as e:
            self.handle_error(e)


# ==================================================
# メモリ応答デモ（改修版）
# ==================================================
class MemoryResponseDemo(BaseDemo):

    @error_handler
    def run(self):

        st.code("""
            messages = self.message_manager.get_default_messages()
            messages.append(
                EasyInputMessageParam(
                    role="user",
                    content=[
                        ResponseInputTextParam(type="input_text", text=question),
                        ResponseInputImageParam(
                            type="input_image",
                            image_url=image_url,
                            detail="auto"
                        ),
                    ],
                )
            )
            response = self.client.responses.create(model=model, input=messages)
        """)
        pass


# ==================================================
# デモマネージャー（改修版）
# ==================================================
class DemoManager:
    """デモの管理クラス（情報パネル機能付き）"""

    def __init__(self):
        self.config = ConfigManager("config.yml")
        self.demos = self._initialize_demos()

    def _initialize_demos(self) -> Dict[str, BaseDemo]:
        """デモインスタンスの初期化"""
        return {
            "Text Responsesサンプル(One Shot)" : TextResponseDemo("responses_One_Shot"),
            "Memory Responsesサンプル(Memory)" : MemoryResponseDemo("responses_memory"),
        }

    def run(self):
        """アプリケーションの実行（情報パネル付き）"""
        UIHelper.init_page()

        # 現在選択中のモデルを取得（デフォルト値）
        # current_model = self.config.get("models.default", "gpt-4o-mini")

        # デモ選択
        demo_name = st.sidebar.radio(
            "デモを選択",
            list(self.demos.keys()),
            key="demo_selection"
        )

        # セッション状態の更新
        if "current_demo" not in st.session_state:
            st.session_state.current_demo = demo_name
        elif st.session_state.current_demo != demo_name:
            st.session_state.current_demo = demo_name

        # 選択されたデモの実行
        demo = self.demos.get(demo_name)
        if demo:
            try:
                demo.run()
            except Exception as e:
                st.error(f"デモの実行中にエラーが発生しました: {str(e)}")
                if st.checkbox("詳細なエラー情報を表示"):
                    st.exception(e)
        else:
            st.error(f"デモ '{demo_name}' が見つかりません")

        # フッター情報
        self._display_footer()

    def _display_footer(self):
        """フッター情報の表示"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 情報")

        # 現在の設定情報
        with st.sidebar.expander("現在の設定"):
            st.json({
                "default_model": self.config.get("models.default"),
                "api_timeout"  : self.config.get("api.timeout"),
                "ui_layout"    : self.config.get("ui.layout"),
            })

        # バージョン情報
        st.sidebar.markdown("### バージョン")
        st.sidebar.markdown("- OpenAI Responses API Demo v2.1 (改修版)")
        st.sidebar.markdown("- Streamlit " + st.__version__)

        # リンク
        st.sidebar.markdown("### リンク")
        st.sidebar.markdown("[OpenAI API ドキュメント](https://platform.openai.com/docs)")
        st.sidebar.markdown("[Streamlit ドキュメント](https://docs.streamlit.io)")


# ==================================================
# メイン関数（改修版）
# ==================================================
def main():
    """アプリケーションのエントリーポイント（改修版）"""

    # (1) ロギングの設定
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # (2) 環境変数のチェック
    if not os.getenv("OPENAI_API_KEY"):
        st.error("環境変数 OPENAI_API_KEY が設定されていません。")
        st.info("export OPENAI_API_KEY='your-api-key' を実行してください。")
        st.stop()

    # (3) セッション状態の初期化
    SessionStateManager.init_session_state()

    # (4) デモマネージャーの作成と実行
    try:
        manager = DemoManager()
        manager.run()
    except Exception as e:
        st.error(f"アプリケーションの起動に失敗しました: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()

# streamlit run a00_00_basic.py --server.port=8501
