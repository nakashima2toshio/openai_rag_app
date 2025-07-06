# helper.py
# ヘルパーモジュール
# -----------------------------------------

# -----------------------------------------
import re
import time
import json
import logging
import yaml
from typing import List, Dict, Any, Optional, Union, Tuple, Literal, Callable
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib

import streamlit as st
import tiktoken
from openai import OpenAI

# -----------------------------------------------------
# "user": ユーザーからのメッセージ
# "assistant": AIアシスタントからの応答
# "system": システムプロンプト（ChatCompletions APIで使用）
# "developer": 開発者による指示（Responses APIで使用）
# -----------------------------------------------------
# [API] responses.createの場合 Messages
# -----------------------------------------------------
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    Response
)
# -----------------------------------------------------
# [API] chat.completions.create の場合のinput
# -----------------------------------------------------
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Case: For Software-Developer
# --------------------------------------------------
developer_text = (
    "You are a strong developer and good at teaching software developer professionals "
    "please provide an up-to-date, informed overview of the API by function, then show "
    "cookbook programs for each, and explain the API options."
    "あなたは強力な開発者でありソフトウェア開発者の専門家に教えるのが得意です。"
    "OpenAIのAPIを機能別に最新かつ詳細に説明してください。"
    "それぞれのAPIのサンプルプログラムを示しAPIのオプションについて説明してください。"
)
user_text = (
    "Organize and identify the problem and list the issues. "
    "Then, provide a solution procedure for the issues you have organized and identified, "
    "and solve the problems/issues according to the solution procedures."
    "不具合、問題を特定し、整理して箇条書きで列挙・説明してください。"
    "次に、整理・特定した問題点の解決手順を示しなさい。"
    "次に、解決手順に従って問題・課題を解決してください。"
)
assistant_text = "OpenAIのAPIを使用するには、公式openaiライブラリが便利です。回答は日本語で"

# ==================================================
# 設定管理
# ==================================================
class ConfigManager:
    # 設定ファイルの管理

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._cache = {}

    def _load_config(self) -> Dict[str, Any]:
        # 設定ファイルの読み込み
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗: {e}")
                return self._get_default_config()
        else:
            logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
            return self._get_default_config()

    @staticmethod
    def _get_default_config(self) -> Dict[str, Any]:
        # デフォルト設定
        return {
            "models": {
                "default"  : "gpt-4o-mini",
                "available": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
            },
            "api"   : {
                "timeout"    : 30,
                "max_retries": 3
            },
            "ui"    : {
                "page_title": "OpenAI API Demo",
                "layout"    : "wide"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        # 設定値の取得（キャッシュ付き）
        if key in self._cache:
            return self._cache[key]

        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = default
                break

        result = value if value is not None else default
        self._cache[key] = result
        return result

    def reload(self):
        # 設定の再読み込み
        self._config = self._load_config()
        self._cache.clear()


# グローバル設定インスタンス
config = ConfigManager()


# ==================================================
# デコレータ
# ==================================================
def error_handler(func):
    # エラーハンドリングデコレータ

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            error_msg = config.get("error_messages.general_error", f"エラーが発生しました: {str(e)}")
            st.error(error_msg)
            if config.get("experimental.debug_mode", False):
                st.exception(e)
            return None

    return wrapper


def timer(func):
    # 実行時間計測デコレータ

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")

        # パフォーマンスモニタリングが有効な場合
        if config.get("experimental.performance_monitoring", True):
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = []
            st.session_state.performance_metrics.append({
                'function'      : func.__name__,
                'execution_time': execution_time,
                'timestamp'     : datetime.now()
            })

        return result

    return wrapper


def cache_result(ttl: int = None):
    # 結果をキャッシュするデコレータ

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            # キャッシュキーの生成
            cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"

            # セッションステートにキャッシュ領域を確保
            if 'cache' not in st.session_state:
                st.session_state.cache = {}

            # キャッシュの確認
            if cache_key in st.session_state.cache:
                cached_data = st.session_state.cache[cache_key]
                if time.time() - cached_data['timestamp'] < (ttl or config.get("cache.ttl", 3600)):
                    return cached_data['result']

            # 関数実行とキャッシュ保存
            result = func(*args, **kwargs)
            st.session_state.cache[cache_key] = {
                'result'   : result,
                'timestamp': time.time()
            }

            # キャッシュサイズ制限
            max_size = config.get("cache.max_size", 100)
            if len(st.session_state.cache) > max_size:
                # 最も古いエントリを削除
                oldest_key = min(st.session_state.cache, key=lambda k: st.session_state.cache[k]['timestamp'])
                del st.session_state.cache[oldest_key]

            return result

        return wrapper

    return decorator


# ==================================================
# メッセージ管理
# chat.completions.crete --- "system", "user", "assistant"
# responses.create       --- "developer", "user", "assistant"
# ==================================================
# Role型の定義
RoleType = Literal["user", "assistant", "system", "developer"]


class MessageManager:
    # メッセージ履歴の管理

    def __init__(self, session_key: str = "message_history"):
        self.session_key = session_key
        self._initialize_messages()

    def _initialize_messages(self):
        # メッセージ履歴の初期化
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self.get_default_messages()

    @staticmethod
    def get_default_messages() -> List[EasyInputMessageParam]:
        # デフォルトメッセージの取得"
        messages = config.get("default_messages", {})
        return [
            EasyInputMessageParam(
                role="developer",
                content=messages.get("developer", "You are a helpful assistant.")
            ),
            EasyInputMessageParam(
                role="user",
                content=messages.get("user", "Hello.")
            ),
            EasyInputMessageParam(
                role="assistant",
                content=messages.get("assistant", "How can I help you today?")
            ),
        ]

    def add_message(self, role: RoleType, content: str):
        # メッセージの追加
        # 型チェックを追加
        valid_roles: List[RoleType] = ["user", "assistant", "system", "developer"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        st.session_state[self.session_key].append(
            EasyInputMessageParam(role=role, content=content)
        )

        # メッセージ数制限
        limit = config.get("ui.message_display_limit", 50)
        if len(st.session_state[self.session_key]) > limit:
            # 最初のdeveloperメッセージは保持
            developer_msg = st.session_state[self.session_key][0] if st.session_state[self.session_key][0][
                                                                         'role'] == 'developer' else None
            st.session_state[self.session_key] = st.session_state[self.session_key][-limit:]
            if developer_msg and st.session_state[self.session_key][0]['role'] != 'developer':
                st.session_state[self.session_key].insert(0, developer_msg)

    def get_messages(self) -> List[EasyInputMessageParam]:
        # メッセージ履歴の取得
        return st.session_state[self.session_key]

    def clear_messages(self):
        # メッセージ履歴のクリア
        st.session_state[self.session_key] = self.get_default_messages()

    def export_messages(self) -> Dict[str, Any]:
        # メッセージ履歴のエクスポート
        return {
            'messages'   : self.get_messages(),
            'exported_at': datetime.now().isoformat()
        }

    def import_messages(self, data: Dict[str, Any]):
        """メッセージ履歴のインポート"""
        if 'messages' in data:
            st.session_state[self.session_key] = data['messages']


# ==================================================
# トークン管理（拡張版）
# ==================================================
class TokenManager:
    """トークン数の管理（新モデル対応）"""

    # モデル別のエンコーディング対応表
    MODEL_ENCODINGS = {
        "gpt-4o"                   : "cl100k_base",
        "gpt-4o-mini"              : "cl100k_base",
        "gpt-4o-audio-preview"     : "cl100k_base",
        "gpt-4o-mini-audio-preview": "cl100k_base",
        "gpt-4.1"                  : "cl100k_base",
        "gpt-4.1-mini"             : "cl100k_base",
        "o1"                       : "cl100k_base",
        "o1-mini"                  : "cl100k_base",
        "o3"                       : "cl100k_base",
        "o3-mini"                  : "cl100k_base",
        "o4"                       : "cl100k_base",
        "o4-mini"                  : "cl100k_base",
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = None) -> int:
        # テキストのトークン数をカウント
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            # モデル別のエンコーディングを取得
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"トークンカウントエラー: {e}")
            # 簡易的な推定（1文字 = 0.5トークン）
            return len(text) // 2

    @classmethod
    def truncate_text(cls, text: str, max_tokens: int, model: str = None) -> str:
        """テキストを指定トークン数に切り詰め"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception as e:
            logger.error(f"テキスト切り詰めエラー: {e}")
            # 簡易的な切り詰め
            estimated_chars = max_tokens * 2
            return text[:estimated_chars]

    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """API使用コストの推定"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        # 設定ファイルから料金を取得
        pricing = config.get("model_pricing", {})
        model_pricing = pricing.get(model, pricing.get("gpt-4o-mini"))

        if not model_pricing:
            # デフォルト料金
            model_pricing = {"input": 0.00015, "output": 0.0006}

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルのトークン制限を取得"""
        limits = {
            "gpt-4o"      : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4o-mini" : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1"     : {"max_tokens": 128000, "max_output": 4096},
            "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
            "o1"          : {"max_tokens": 128000, "max_output": 32768},
            "o1-mini"     : {"max_tokens": 128000, "max_output": 65536},
            "o3"          : {"max_tokens": 200000, "max_output": 100000},
            "o3-mini"     : {"max_tokens": 200000, "max_output": 100000},
            "o4"          : {"max_tokens": 256000, "max_output": 128000},
            "o4-mini"     : {"max_tokens": 256000, "max_output": 128000},
        }
        return limits.get(model, {"max_tokens": 128000, "max_output": 4096})


# ==================================================
# UI ヘルパー（拡張版）
# ==================================================
class UIHelper:
    # Streamlit UI用のヘルパー関数（拡張版）

    @staticmethod
    def init_page(title: str = None, sidebar_title: str = None):
        # ページの初期化
        if title is None:
            title = config.get("ui.page_title", "OpenAI API Demo")
        if sidebar_title is None:
            sidebar_title = "メニュー"

        st.set_page_config(
            page_title=title,
            page_icon=config.get("ui.page_icon", "🤖"),
            layout=config.get("ui.layout", "wide")
        )

        st.header(title)
        st.sidebar.title(sidebar_title)

    # -----------------------------------------------------------
    # select model:
    # -----------------------------------------------------------
    @staticmethod
    def select_model(key: str = "model_selection", category: str = None) -> str:
        # モデル選択UI（カテゴリ対応）
        models = config.get("models.available", ["gpt-4o", "gpt-4o-mini"])
        default_model = config.get("models.default", "gpt-4o-mini")

        # カテゴリでフィルタリング
        if category:
            if category == "reasoning":
                models = [m for m in models if m.startswith("o")]
            elif category == "standard":
                models = [m for m in models if m.startswith("gpt")]
            elif category == "audio":
                models = [m for m in models if "audio" in m]

        default_index = models.index(default_model) if default_model in models else 0

        selected = st.sidebar.selectbox(
            "モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("モデル情報"):
            limits = TokenManager.get_model_limits(selected)
            st.write(f"最大入力: {limits['max_tokens']:,} tokens")
            st.write(f"最大出力: {limits['max_output']:,} tokens")

        return selected

    # -----------------------------------------------------------
    # select speech model (Text-to-Speech):
    # -----------------------------------------------------------
    @staticmethod
    def select_speech_model(key: str = "speech_model_selection", category: str = None) -> str:
        # 音声合成モデル選択UI（カテゴリ対応）
        # 音声合成用モデルリスト
        all_speech_models = [
            "tts-1", "tts-1-hd",  # 専用TTS
            "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview",  # 音声対話
            "o3-mini", "o4-mini", "o1-mini"  # 推論系（音声対応）
        ]

        # デフォルトモデル
        default_speech_model = "tts-1"

        # カテゴリでフィルタリング
        if category:
            if category == "tts":
                models = [m for m in all_speech_models if m.startswith("tts")]
            elif category == "audio_chat":
                models = [m for m in all_speech_models if "audio" in m]
            elif category == "reasoning":
                models = [m for m in all_speech_models if m.startswith("o")]
            else:
                models = all_speech_models
        else:
            models = all_speech_models

        # デフォルトインデックスの設定
        default_index = models.index(default_speech_model) if default_speech_model in models else 0

        selected = st.sidebar.selectbox(
            "音声合成モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("音声モデル情報"):
            if selected.startswith("tts"):
                st.write("**TTS専用モデル**")
                if selected == "tts-1":
                    st.write("- 高速・低コスト")
                    st.write("- 音質: 標準")
                elif selected == "tts-1-hd":
                    st.write("- 高音質・低遅延")
                    st.write("- 音質: 高品質")
            elif "audio" in selected:
                st.write("**音声対話モデル**")
                st.write("- テキスト+音声入出力対応")
                st.write("- リアルタイム対話可能")
                # OpenAI APIの制限情報があれば表示
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")
            elif selected.startswith("o"):
                st.write("**推論系モデル（音声対応）**")
                st.write("- 高度な推論能力")
                st.write("- 複雑なタスクに対応")
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")

        return selected

    # -----------------------------------------------------------
    # select whisper model (Speech-to-Text):
    # -----------------------------------------------------------
    @staticmethod
    def select_whisper_model(key: str = "whisper_model_selection", category: str = None) -> str:
        # 音声認識/翻訳モデル選択UI（カテゴリ対応）
        # 音声認識/翻訳用モデルリスト
        all_whisper_models = [
            "whisper-1",  # 専用STT
            "gpt-4o-transcribe", "gpt-4o-mini-transcribe",  # GPT系STT
            "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview"  # 音声対話（STT機能含む）
        ]

        # デフォルトモデル
        default_whisper_model = "whisper-1"

        # カテゴリでフィルタリング
        if category:
            if category == "whisper":
                models = [m for m in all_whisper_models if "whisper" in m]
            elif category == "transcribe":
                models = [m for m in all_whisper_models if "transcribe" in m]
            elif category == "audio_chat":
                models = [m for m in all_whisper_models if "audio-preview" in m]
            elif category == "gpt":
                models = [m for m in all_whisper_models if m.startswith("gpt")]
            else:
                models = all_whisper_models
        else:
            models = all_whisper_models

        # デフォルトインデックスの設定
        default_index = models.index(default_whisper_model) if default_whisper_model in models else 0

        selected = st.sidebar.selectbox(
            "音声認識/翻訳モデルを選択",
            models,
            index=default_index,
            key=key
        )

        # モデル情報の表示
        with st.sidebar.expander("音声認識モデル情報"):
            if selected == "whisper-1":
                st.write("**Whisper専用モデル**")
                st.write("- 多言語対応")
                st.write("- 転写・翻訳対応")
                st.write("- ファイルサイズ: 最大25MB")
                st.write("- 対応形式: mp3, mp4, wav, webm, m4a, flac, etc.")
            elif "transcribe" in selected:
                st.write("**GPT系転写モデル**")
                st.write("- 高精度転写")
                st.write("- コンテキスト理解")
                if "mini" in selected:
                    st.write("- 高速・低コスト版")
                else:
                    st.write("- 高性能版")
            elif "audio-preview" in selected:
                st.write("**音声対話モデル（STT機能）**")
                st.write("- リアルタイム音声処理")
                st.write("- テキスト+音声入出力")
                limits = TokenManager.get_model_limits(selected)
                st.write(f"最大入力: {limits['max_tokens']:,} tokens")
                st.write(f"最大出力: {limits['max_output']:,} tokens")

            # 共通情報
            st.write("---")
            st.write("**対応言語**: 日本語、英語、その他多数")

            # コスト情報（設定ファイルから取得できる場合）
            pricing = config.get("model_pricing", {}).get(selected)
            if pricing:
                st.write("**料金情報**:")
                if "input" in pricing:
                    st.write(f"- 入力: ${pricing['input']}/1K tokens")
                if "output" in pricing:
                    st.write(f"- 出力: ${pricing['output']}/1K tokens")

        return selected

    # -----------------------------------------------------------
    @staticmethod
    def create_form(key: str, submit_label: str = "送信") -> Tuple[Any, bool]:
        """フォームの作成"""
        form = st.form(key=key)
        submitted = form.form_submit_button(submit_label)
        return form, submitted

    @staticmethod
    def display_messages(messages: List[EasyInputMessageParam], show_system: bool = False):
        # メッセージ履歴の表示（改良版）
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                with st.chat_message("user"):
                    if isinstance(content, list):
                        # マルチモーダルコンテンツの処理
                        for item in content:
                            if item.get("type") == "input_text":
                                st.markdown(item.get("text", ""))
                            elif item.get("type") == "input_image":
                                st.image(item.get("image_url", ""))
                    else:
                        st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)
            elif (role == "developer" or role == "system") and show_system:
                with st.expander(f"{role.capitalize()} Message", expanded=False):
                    st.markdown(f"*{content}*")

    @staticmethod
    def show_token_info(text: str, model: str = None):
        """トークン情報の表示（拡張版）"""
        token_count = TokenManager.count_tokens(text, model)
        limits = TokenManager.get_model_limits(model)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("トークン数", f"{token_count:,}")
        with col2:
            usage_percent = (token_count / limits['max_tokens']) * 100
            st.metric("使用率", f"{usage_percent:.1f}%")

        # コスト推定（仮定: 出力は入力の50%）
        estimated_output = token_count // 2
        cost = TokenManager.estimate_cost(token_count, estimated_output, model)
        st.sidebar.metric("推定コスト", f"${cost:.4f}")

        # プログレスバー
        st.sidebar.progress(min(usage_percent / 100, 1.0))

    @staticmethod
    def create_tabs(tab_names: List[str], key: str = "tabs") -> List[Any]:
        # タブの作成
        return st.tabs(tab_names)

    @staticmethod
    def create_columns(spec: List[Union[int, float]], gap: str = "medium") -> List[Any]:
        # カラムの作成
        return st.columns(spec, gap=gap)

    @staticmethod
    def show_metrics(metrics: Dict[str, Any], columns: int = 3):
        # メトリクスの表示
        cols = st.columns(columns)
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(value, dict):
                    st.metric(label, value.get('value'), value.get('delta'))
                else:
                    st.metric(label, value)

    @staticmethod
    def create_download_button(data: Any, filename: str, mime_type: str = "text/plain", label: str = "ダウンロード"):
        # ダウンロードボタンの作成
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False, indent=2)
        elif isinstance(data, list):
            data = json.dumps(data, ensure_ascii=False, indent=2)

        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime=mime_type
        )


# ==================================================
# レスポンス処理
# ==================================================
class ResponseProcessor:
    # API レスポンスの処理

    @staticmethod
    def extract_text(response: Response) -> List[str]:
        # レスポンスからテキストを抽出"""
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

    @staticmethod
    def format_response(response: Response) -> Dict[str, Any]:
        """レスポンスを整形"""
        return {
            "id"        : getattr(response, "id", None),
            "model"     : getattr(response, "model", None),
            "created_at": getattr(response, "created_at", None),
            "text"      : ResponseProcessor.extract_text(response),
            "usage"     : getattr(response, "usage", {}),
        }

    @staticmethod
    def display_response(response: Response, show_details: bool = True):
        """レスポンスの表示（改良版）"""
        texts = ResponseProcessor.extract_text(response)

        if texts:
            for i, text in enumerate(texts, 1):
                if len(texts) > 1:
                    st.subheader(f"回答 {i}")
                st.write(text)
        else:
            st.warning("テキストが見つかりませんでした")

        # 詳細情報の表示
        if show_details:
            with st.expander("詳細情報"):
                formatted = ResponseProcessor.format_response(response)

                # 使用状況の表示
                if 'usage' in formatted and formatted['usage']:
                    usage = formatted['usage']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("入力トークン", usage.get('prompt_tokens', 0))
                    with col2:
                        st.metric("出力トークン", usage.get('completion_tokens', 0))
                    with col3:
                        st.metric("合計トークン", usage.get('total_tokens', 0))

                # JSON形式での表示
                st.json(formatted)

    @staticmethod
    def save_response(response: Response, filename: str = None) -> str:
        # レスポンスの保存
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.json"

        formatted = ResponseProcessor.format_response(response)

        # ファイルパスの生成
        logs_dir = Path(config.get("paths.logs_dir", "logs"))
        logs_dir.mkdir(exist_ok=True)
        filepath = logs_dir / filename

        # 保存
        save_json_file(formatted, str(filepath))

        return str(filepath)


# ==================================================
# 基底クラス
# ==================================================
class DemoBase(ABC):
    # デモの基底クラス

    def __init__(self, demo_name: str, title: str = None):
        self.demo_name = demo_name
        self.title = title or demo_name
        self.key_prefix = sanitize_key(demo_name)
        self.message_manager = MessageManager(f"messages_{self.key_prefix}")

    @abstractmethod
    def run(self):
        # デモの実行（サブクラスで実装）
        pass

    def setup_ui(self):
        # 共通UI設定
        st.subheader(self.title)

        # モデル選択
        self.model = UIHelper.select_model(f"model_{self.key_prefix}")

        # メッセージ履歴のクリア
        if st.sidebar.button("履歴クリア", key=f"clear_{self.key_prefix}"):
            self.message_manager.clear_messages()
            st.rerun()

    def display_messages(self):
        # メッセージの表示
        messages = self.message_manager.get_messages()
        UIHelper.display_messages(messages)

    def add_user_message(self, content: str):
        # ユーザーメッセージの追加
        self.message_manager.add_message("user", content)

    def add_assistant_message(self, content: str):
        # アシスタントメッセージの追加
        self.message_manager.add_message("assistant", content)

    @error_handler
    @timer
    def call_api(self, messages: List[EasyInputMessageParam], **kwargs) -> Response:
        # API呼び出し（共通処理）
        client = OpenAI()

        # デフォルトパラメータ
        params = {
            "model"     : self.model,
            "input"  : messages,
            # "max_tokens": config.get("api.max_tokens", 4096),
        }
        params.update(kwargs)

        # API呼び出し
        response = client.responses.create(**params)

        return response


# ==================================================
# ユーティリティ関数
# ==================================================
def sanitize_key(name: str) -> str:
    # Streamlit key用に安全な文字列へ変換
    return re.sub(r'[^0-9a-zA-Z_]', '_', name).lower()


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    # JSONファイルの読み込み
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSONファイル読み込みエラー: {e}")
        return None


def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    # JSONファイルの保存
    try:
        # ディレクトリの作成
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"JSONファイル保存エラー: {e}")
        return False


def format_timestamp(timestamp: Union[int, float, str] = None) -> str:
    # タイムスタンプのフォーマット
    if timestamp is None:
        timestamp = time.time()

    if isinstance(timestamp, str):
        return timestamp

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def create_session_id() -> str:
    # セッションIDの生成
    return hashlib.md5(f"{time.time()}_{id(st)}".encode()).hexdigest()[:8]


# ==================================================
# 後方互換性のための関数
# ==================================================
def init_page(title: str):
    # 後方互換性のための関数
    UIHelper.init_page(title)


def init_messages(demo_name: str = ""):
    # 後方互換性のための関数
    manager = MessageManager(f"messages_{sanitize_key(demo_name)}")

    if st.sidebar.button("会話履歴のクリア", key=f"clear_{sanitize_key(demo_name)}"):
        manager.clear_messages()


def select_model(demo_name: str = "") -> str:
    # 後方互換性のための関数
    return UIHelper.select_model(f"model_{sanitize_key(demo_name)}")


def get_default_messages() -> List[EasyInputMessageParam]:
    # 後方互換性のための関数
    manager = MessageManager()
    return manager.get_default_messages()


def extract_text_from_response(response: Response) -> List[str]:
    # 後方互換性のための関数
    return ResponseProcessor.extract_text(response)


def append_user_message(append_text: str, image_url: Optional[str] = None) -> List[EasyInputMessageParam]:
    # 後方互換性のための関数
    messages = get_default_messages()
    if image_url:
        content = [
            ResponseInputTextParam(type="input_text", text=append_text),
            ResponseInputImageParam(type="input_image", image_url=image_url, detail="auto")
        ]
        messages.append(EasyInputMessageParam(role="user", content=content))
    else:
        messages.append(EasyInputMessageParam(role="user", content=append_text))
    return messages


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    # クラス
    'ConfigManager',
    'MessageManager',
    'TokenManager',
    'UIHelper',
    'ResponseProcessor',
    'DemoBase',

    # 型定義
    'RoleType',

    # デコレータ
    'error_handler',
    'timer',
    'cache_result',

    # ユーティリティ
    'sanitize_key',
    'load_json_file',
    'save_json_file',
    'format_timestamp',
    'create_session_id',

    # 後方互換性
    'init_page',
    'init_messages',
    'select_model',
    'get_default_messages',
    'extract_text_from_response',
    'append_user_message',
]
