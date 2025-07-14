#
# 代表的なMCPサーバー一覧
# 公式サーバー（Anthropic/OpenAI提供）
# ------------------------------------------------
# File System - ローカルファイルシステムへのアクセス
# GitHub - GitHubリポジトリ管理とAPI統合
# Google Drive - Google Driveファイルアクセスと検索
# Slack - Slackワークスペース統合とメッセージング
# PostgreSQL - データベースアクセス（読み取り専用）
# Puppeteer - ブラウザ自動化とWebスクレイピング
# Git - Gitリポジトリの読み取り・検索・操作
# Google Maps - 位置情報サービス、ルート検索
# Brave Search - Brave検索APIを使用したWeb検索
# SQLite - SQLiteデータベース操作
# GitHub - wong2/awesome-mcp-servers: A curated list of Model Context Protocol (MCP) servers
# ------------------------------------------------
# サードパーティ製人気サーバー
# ------------------------------------------------
# Docker - Dockerコンテナ管理
# Notion - Notionページとデータベース操作
# Spotify - Spotify API統合（音楽制御）
# Microsoft 365 - Office、Outlook、Excel統合
# Raygun - エラートラッキングと監視
# Cloudflare - エッジコンピューティングとCDN
# Redis - Redisキーバリューストア操作
# MySQL - MySQLデータベース統合
# Google Sheets - Googleスプレッドシート操作
# Email (SMTP) - メール送信機能
# HuggingfaceGitHub
# ------------------------------------------------
"""
OpenAI Responses API + MCP (Model Context Protocol) サーバー利用例 - 修正版
=================================================================

このファイルは、OpenAI Responses APIでMCPサーバーを利用する
様々なパターンのコード例を提供します。

前提条件:
- OpenAI API Key (OPENAI_API_KEY環境変数)
- 各MCPサーバーのAPI Key（必要に応じて）
- OpenAI Python SDK version 1.50.0以上

修正内容:
- messages → input パラメータに変更
- MCPツール設定にserver_labelとrequire_approvalを追加
- レスポンス処理を新しいAPI仕様に対応
"""
"""
OpenAI API + MCP (Model Context Protocol) 完全型対応版
====================================================

このファイルは、OpenAI APIの適切な型付きオブジェクトを使用して
MCPサーバーを利用する完全なコード例を提供します。

前提条件:
- OpenAI API Key (OPENAI_API_KEY環境変数)
- OpenAI Python SDK version 1.60.0以上
- 各MCPサーバーのAPI Key（必要に応じて）

型対応内容:
- すべてのツールで適切な型付きオブジェクトを使用
- HostedMCPTool、WebSearchTool等の正式な型定義
- 型チェックエラーの完全解決
- Pydantic BaseModelを使用した型安全な実装
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

# OpenAI関連のインポート
from openai import OpenAI, AsyncOpenAI

# 型定義のインポート
try:
    # OpenAI Responses API用の型
    from openai.types.responses import (
        ResponseCreateParams,
        Response as OpenAIResponse
    )

    # 組み込みツール型のインポート
    from openai.types.responses.create_params import (
        Tool as ResponseTool,
        MCPTool,
        CodeInterpreterTool,
        WebSearchPreviewTool,
        ImageGenerationTool
    )

    TYPES_AVAILABLE = True
    print("✅ OpenAI型定義が利用可能です")

except ImportError:
    print("⚠️ OpenAI型定義をインポートできません。fallback実装を使用します")
    TYPES_AVAILABLE = False
    ResponseTool = Dict[str, Any]
    MCPTool = Dict[str, Any]
    CodeInterpreterTool = Dict[str, Any]
    WebSearchPreviewTool = Dict[str, Any]
    ImageGenerationTool = Dict[str, Any]

# Pydantic for data validation
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    print("⚠️ Pydanticが利用できません")
    PYDANTIC_AVAILABLE = False
    BaseModel = object

# OpenAIクライアントの初期化
client = OpenAI()
async_client = AsyncOpenAI()


# ================================================
# 型定義とデータクラス
# ================================================

@dataclass
class MCPServerConfig:
    """MCPサーバー設定の型安全なデータクラス"""
    server_url: str
    server_label: str
    headers: Optional[Dict[str, str]] = None
    allowed_tools: Optional[List[str]] = None
    require_approval: str = "never"

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.allowed_tools is None:
            self.allowed_tools = []


@dataclass
class ProjectConfig:
    """プロジェクト設定"""
    base_dir: Path
    this_dir: Path
    datasets_dir: Path
    helper_dir: Path
    page_title: str = "OpenAI API + MCP Learning"
    page_icon: str = "🤖"
    default_model: str = "gpt-4o"
    available_models: List[str] = None

    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"
            ]
        self.datasets_dir.mkdir(exist_ok=True, parents=True)


# プロジェクト設定の初期化
BASE_DIR = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent

config = ProjectConfig(
    base_dir=BASE_DIR,
    this_dir=THIS_DIR,
    datasets_dir=BASE_DIR / 'datasets',
    helper_dir=BASE_DIR / 'a0_common_helper'
)


# ================================================
# 型安全なツールファクトリー
# ================================================

class TypedToolFactory:
    """型安全なツール作成ファクトリークラス"""

    @staticmethod
    def create_mcp_tool(config: MCPServerConfig) -> Union[MCPTool, Dict[str, Any]]:
        """型安全なMCPツールを作成"""
        if TYPES_AVAILABLE:
            # 正式な型を使用
            return MCPTool(
                type="mcp",
                server_label=config.server_label,
                server_url=config.server_url,
                headers=config.headers,
                allowed_tools=config.allowed_tools,
                require_approval=config.require_approval
            )
        else:
            # フォールバック: 辞書形式
            return {
                "type"            : "mcp",
                "server_label"    : config.server_label,
                "server_url"      : config.server_url,
                "headers"         : config.headers,
                "allowed_tools"   : config.allowed_tools,
                "require_approval": config.require_approval
            }

    @staticmethod
    def create_code_interpreter_tool() -> Union[CodeInterpreterTool, Dict[str, Any]]:
        """型安全なコードインタープリターツールを作成"""
        if TYPES_AVAILABLE:
            return CodeInterpreterTool(type="code_interpreter")
        else:
            return {"type": "code_interpreter"}

    @staticmethod
    def create_web_search_tool() -> Union[WebSearchPreviewTool, Dict[str, Any]]:
        """型安全なWeb検索ツールを作成"""
        if TYPES_AVAILABLE:
            return WebSearchPreviewTool(type="web_search_preview")
        else:
            return {"type": "web_search_preview"}

    @staticmethod
    def create_image_generation_tool() -> Union[ImageGenerationTool, Dict[str, Any]]:
        """型安全な画像生成ツールを作成"""
        if TYPES_AVAILABLE:
            return ImageGenerationTool(type="image_generation")
        else:
            return {"type": "image_generation"}


# ================================================
# 基底クラス（型安全版）
# ================================================

class BaseTypedDemo(ABC):
    """型安全なデモ機能の基底クラス"""

    def __init__(self, demo_name: str):
        self.demo_name = demo_name
        self.client = client
        self.async_client = async_client
        self.tool_factory = TypedToolFactory()

    def validate_environment(self) -> List[str]:
        """環境設定の検証"""
        issues = []
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not set")
        return issues

    @abstractmethod
    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """このデモで使用するツールを作成"""
        pass

    @abstractmethod
    def get_input_message(self) -> str:
        """入力メッセージを取得"""
        pass

    def run_sync(self) -> OpenAIResponse:
        """同期実行"""
        tools = self.create_tools()
        input_message = self.get_input_message()

        try:
            response = self.client.responses.create(
                model=config.default_model,
                input=input_message,
                tools=tools
            )
            return response
        except Exception as e:
            print(f"Error in {self.demo_name}: {e}")
            raise

    async def run_async(self) -> OpenAIResponse:
        """非同期実行"""
        tools = self.create_tools()
        input_message = self.get_input_message()

        try:
            response = await self.async_client.responses.create(
                model=config.default_model,
                input=input_message,
                tools=tools
            )
            return response
        except Exception as e:
            print(f"Error in {self.demo_name}: {e}")
            raise


# ================================================
# 具体的なデモ実装（型安全版）
# ================================================

class GitHubMCPDemo(BaseTypedDemo):
    """GitHub MCPサーバーを使用するデモ（型安全版）"""

    def __init__(self):
        super().__init__("GitHub MCP Demo (Typed)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """GitHub MCPツールを作成"""
        github_config = MCPServerConfig(
            server_url="https://github-mcp-server.example.com",
            server_label="github",
            headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            allowed_tools=[
                "search_repository",
                "get_file_content",
                "list_files",
                "get_commit_info"
            ],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(github_config)
        ]

    def get_input_message(self) -> str:
        return "Please search for Python files in my repository and show me the main functions and classes."


class MultiMCPDemo(BaseTypedDemo):
    """複数MCPサーバーを使用するデモ（型安全版）"""

    def __init__(self):
        super().__init__("Multi MCP Demo (Typed)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """複数のMCPツールを作成"""
        # GitHub MCP設定
        github_config = MCPServerConfig(
            server_url="https://github-mcp-server.example.com",
            server_label="github",
            headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            allowed_tools=[
                "get_recent_commits",
                "get_file_content",
                "get_commit_diff"
            ],
            require_approval="never"
        )

        # Slack MCP設定
        slack_config = MCPServerConfig(
            server_url="https://slack-mcp-server.example.com",
            server_label="slack",
            headers={"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"},
            allowed_tools=[
                "post_message",
                "list_channels",
                "get_channel_history"
            ],
            require_approval="never"
        )

        # Web検索ツールも追加
        return [
            self.tool_factory.create_mcp_tool(github_config),
            self.tool_factory.create_mcp_tool(slack_config),
            self.tool_factory.create_web_search_tool()
        ]

    def get_input_message(self) -> str:
        return """
        Please follow these steps:
        1. Use GitHub MCP to examine recent commits in the repository
        2. Analyze the code for potential issues or improvements
        3. Use web search if you need additional information about best practices
        4. Use Slack MCP to post a summary to the #code-review channel

        Focus on: security issues, performance problems, and code quality.
        """


class DatabaseAnalysisDemo(BaseTypedDemo):
    """データベース分析デモ（型安全版）"""

    def __init__(self):
        super().__init__("Database Analysis Demo (Typed)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """データベース分析用ツールを作成"""
        # PostgreSQL MCP設定
        db_config = MCPServerConfig(
            server_url="https://postgresql-mcp.example.com",
            server_label="postgresql",
            headers={"Authorization": f"Bearer {os.getenv('DB_API_KEY')}"},
            allowed_tools=[
                "execute_query",
                "get_schema",
                "list_tables",
                "describe_table"
            ],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(db_config),
            self.tool_factory.create_code_interpreter_tool()
        ]

    def get_input_message(self) -> str:
        return """
        You are a data analyst. Please:
        1. Examine the database schema using PostgreSQL MCP
        2. Query sales data from the last 30 days
        3. Calculate key metrics (total sales, average order value, top products) using code interpreter
        4. Create visualizations to show trends
        5. Provide insights and recommendations

        Use proper SQL practices and explain your analysis process.
        """


class FileManagementDemo(BaseTypedDemo):
    """ファイル管理デモ（型安全版）"""

    def __init__(self):
        super().__init__("File Management Demo (Typed)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """ファイル管理用ツールを作成"""
        # File System MCP設定
        filesystem_config = MCPServerConfig(
            server_url="http://localhost:8000/filesystem",
            server_label="filesystem",
            headers={},
            allowed_tools=[
                "list_files",
                "read_file",
                "create_directory",
                "get_file_info"
            ],
            require_approval="always"  # ファイル操作は承認必須
        )

        # Google Drive MCP設定
        gdrive_config = MCPServerConfig(
            server_url="https://gdrive-mcp.example.com",
            server_label="gdrive",
            headers={"Authorization": f"Bearer {os.getenv('GOOGLE_DRIVE_TOKEN')}"},
            allowed_tools=[
                "upload_file",
                "create_folder",
                "search_files",
                "get_file_metadata"
            ],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(filesystem_config),
            self.tool_factory.create_mcp_tool(gdrive_config)
        ]

    def get_input_message(self) -> str:
        return """
        You are a file management assistant. Please help the user:
        1. Organize local files by type and date
        2. Backup important documents to Google Drive
        3. Create a summary report of the file organization

        Always ask for confirmation before moving or deleting files.
        The user wants to organize their Downloads folder and backup documents.
        """


class WebScrapingAnalysisDemo(BaseTypedDemo):
    """Webスクレイピング分析デモ（型安全版）"""

    def __init__(self):
        super().__init__("Web Scraping Analysis Demo (Typed)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """Webスクレイピング分析用ツールを作成"""
        # Puppeteer MCP設定
        puppeteer_config = MCPServerConfig(
            server_url="https://puppeteer-mcp.example.com",
            server_label="puppeteer",
            headers={},
            allowed_tools=[
                "navigate_to_page",
                "extract_text",
                "take_screenshot",
                "wait_for_element",
                "get_page_content"
            ],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(puppeteer_config),
            self.tool_factory.create_web_search_tool(),
            self.tool_factory.create_code_interpreter_tool()
        ]

    def get_input_message(self) -> str:
        return """
        You are a market research analyst. Please:
        1. Use web search to find competitor websites selling yoga pants
        2. Use Puppeteer MCP to scrape product pricing data from competitor websites
        3. Use code interpreter to analyze price trends and patterns
        4. Generate a competitive analysis report with visualizations

        Focus on accuracy and provide data sources for all findings.
        Research pricing for yoga pants and create a competitive analysis.
        """


class ComprehensiveAnalysisDemo(BaseTypedDemo):
    """包括的分析デモ（すべてのツール型を使用）"""

    def __init__(self):
        super().__init__("Comprehensive Analysis Demo (All Tool Types)")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """すべての種類のツールを作成"""
        # 複数のMCPサーバー設定
        github_config = MCPServerConfig(
            server_url="https://github-mcp-server.example.com",
            server_label="github",
            headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            allowed_tools=["search_repository", "get_file_content"],
            require_approval="never"
        )

        db_config = MCPServerConfig(
            server_url="https://postgresql-mcp.example.com",
            server_label="database",
            headers={"Authorization": f"Bearer {os.getenv('DB_API_KEY')}"},
            allowed_tools=["execute_query", "get_schema"],
            require_approval="never"
        )

        # すべてのツール型を含む
        return [
            self.tool_factory.create_mcp_tool(github_config),
            self.tool_factory.create_mcp_tool(db_config),
            self.tool_factory.create_web_search_tool(),
            self.tool_factory.create_code_interpreter_tool(),
            self.tool_factory.create_image_generation_tool()
        ]

    def get_input_message(self) -> str:
        return """
        Please perform a comprehensive analysis:

        1. Use GitHub MCP to analyze our codebase structure
        2. Use Database MCP to get performance metrics
        3. Use web search to research industry best practices
        4. Use code interpreter to process and visualize the data
        5. Generate charts and diagrams with image generation
        6. Create a comprehensive report with recommendations

        Focus on: code quality, performance, scalability, and comparison with industry standards.
        """


# ================================================
# ストリーミング対応デモ（型安全版）
# ================================================

class TypedStreamingDemo(BaseTypedDemo):
    """型安全なストリーミングデモ"""

    def __init__(self):
        super().__init__("Typed Streaming Demo")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """ストリーミング用ツールを作成"""
        gdrive_config = MCPServerConfig(
            server_url="https://gdrive-mcp.example.com",
            server_label="gdrive",
            headers={"Authorization": f"Bearer {os.getenv('GOOGLE_DRIVE_TOKEN')}"},
            allowed_tools=[
                "search_files",
                "get_file_content",
                "list_recent_files"
            ],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(gdrive_config),
            self.tool_factory.create_web_search_tool()
        ]

    def get_input_message(self) -> str:
        return "Search my Google Drive for recent presentations and summarize their contents. Also search the web for related industry trends."

    async def run_streaming(self) -> str:
        """ストリーミング実行"""
        tools = self.create_tools()
        input_message = self.get_input_message()

        try:
            stream = await self.async_client.responses.create(
                model=config.default_model,
                input=input_message,
                tools=tools,
                stream=True
            )

            full_response = ""
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        if choice.delta.content:
                            content = choice.delta.content
                            full_response += content
                            print(content, end='', flush=True)

            return full_response

        except Exception as e:
            print(f"Streaming error: {e}")
            raise


# ================================================
# カスタム関数ツール（型安全版）
# ================================================

if PYDANTIC_AVAILABLE:
    class CalculationRequest(BaseModel):
        """計算リクエストの型定義"""
        data: List[float] = Field(..., description="計算対象のデータリスト")
        metric_type: str = Field(default="average", description="メトリクスの種類")


    class AnalysisResult(BaseModel):
        """分析結果の型定義"""
        metric_value: float = Field(..., description="計算されたメトリクス値")
        data_count: int = Field(..., description="データ点の数")
        metric_type: str = Field(..., description="使用されたメトリクス種類")


def calculate_metrics_typed(data: List[float], metric_type: str = "average") -> Dict[str, Any]:
    """
    型安全な数値データメトリクス計算関数

    Args:
        data: 計算対象のデータリスト
        metric_type: メトリクスの種類 ("average", "sum", "max", "min")

    Returns:
        計算結果の辞書
    """
    if not data:
        return {"error": "データが空です", "metric_value": 0, "data_count": 0}

    if metric_type == "average":
        value = sum(data) / len(data)
    elif metric_type == "sum":
        value = sum(data)
    elif metric_type == "max":
        value = max(data)
    elif metric_type == "min":
        value = min(data)
    else:
        return {"error": f"未知のメトリクス種類: {metric_type}", "metric_value": 0, "data_count": len(data)}

    return {
        "metric_value": value,
        "data_count"  : len(data),
        "metric_type" : metric_type,
        "success"     : True
    }


# ================================================
# デモマネージャー（型安全版）
# ================================================

class TypedDemoManager:
    """型安全なデモ管理クラス"""

    def __init__(self):
        self.demos: Dict[str, BaseTypedDemo] = {
            "github_mcp"       : GitHubMCPDemo(),
            "multi_mcp"        : MultiMCPDemo(),
            "database_analysis": DatabaseAnalysisDemo(),
            "file_management"  : FileManagementDemo(),
            "web_scraping"     : WebScrapingAnalysisDemo(),
            "comprehensive"    : ComprehensiveAnalysisDemo(),
            "streaming"        : TypedStreamingDemo()
        }

    def add_demo(self, name: str, demo: BaseTypedDemo):
        """新しいデモを追加"""
        self.demos[name] = demo

    def list_demos(self) -> List[str]:
        """利用可能なデモのリストを取得"""
        return list(self.demos.keys())

    def run_demo_sync(self, demo_name: str) -> OpenAIResponse:
        """デモを同期実行"""
        if demo_name not in self.demos:
            raise ValueError(f"Unknown demo: {demo_name}")

        demo = self.demos[demo_name]
        issues = demo.validate_environment()

        if issues:
            print(f"Environment issues for {demo_name}:")
            for issue in issues:
                print(f"  - {issue}")

        return demo.run_sync()

    async def run_demo_async(self, demo_name: str) -> OpenAIResponse:
        """デモを非同期実行"""
        if demo_name not in self.demos:
            raise ValueError(f"Unknown demo: {demo_name}")

        demo = self.demos[demo_name]
        issues = demo.validate_environment()

        if issues:
            print(f"Environment issues for {demo_name}:")
            for issue in issues:
                print(f"  - {issue}")

        return await demo.run_async()

    async def run_streaming_demo(self, demo_name: str = "streaming") -> str:
        """ストリーミングデモを実行"""
        if demo_name not in self.demos:
            raise ValueError(f"Unknown demo: {demo_name}")

        demo = self.demos[demo_name]
        if isinstance(demo, TypedStreamingDemo):
            return await demo.run_streaming()
        else:
            raise ValueError(f"Demo {demo_name} does not support streaming")


# ================================================
# レスポンス処理ユーティリティ（型安全版）
# ================================================

class TypedResponseProcessor:
    """型安全なレスポンス処理クラス"""

    @staticmethod
    def extract_content(response: OpenAIResponse) -> str:
        """レスポンスからコンテンツを抽出"""
        try:
            if hasattr(response, 'output_text') and response.output_text:
                return response.output_text
            elif hasattr(response, 'output') and response.output:
                content_parts = []
                for item in response.output:
                    if hasattr(item, 'content'):
                        for content in item.content:
                            if hasattr(content, 'text'):
                                content_parts.append(content.text)
                            elif hasattr(content, 'type') and content.type == "output_text":
                                if hasattr(content, 'text'):
                                    content_parts.append(content.text)
                return "\n".join(content_parts)
            else:
                return str(response)
        except Exception as e:
            print(f"Content extraction error: {e}")
            return str(response)

    @staticmethod
    def extract_tool_calls(response: OpenAIResponse) -> List[Dict[str, Any]]:
        """レスポンスからツール呼び出し情報を抽出"""
        tool_calls = []
        try:
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type'):
                        if item.type == "mcp_tool_call":
                            tool_calls.append({
                                "type"        : "mcp",
                                "server_label": getattr(item, 'server_label', 'unknown'),
                                "tool_name"   : getattr(item, 'tool_name', 'unknown'),
                                "status"      : getattr(item, 'status', 'unknown')
                            })
                        elif item.type == "web_search_call":
                            tool_calls.append({
                                "type"  : "web_search",
                                "status": getattr(item, 'status', 'unknown')
                            })
                        elif item.type == "code_interpreter_call":
                            tool_calls.append({
                                "type"  : "code_interpreter",
                                "status": getattr(item, 'status', 'unknown')
                            })
        except Exception as e:
            print(f"Tool call extraction error: {e}")

        return tool_calls

    @staticmethod
    def get_usage_info(response: OpenAIResponse) -> Dict[str, Any]:
        """使用情報を取得"""
        usage_info = {}
        try:
            if hasattr(response, 'usage'):
                usage_info = {
                    "input_tokens" : getattr(response.usage, 'input_tokens', 0),
                    "output_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens" : getattr(response.usage, 'total_tokens', 0)
                }

            if hasattr(response, 'model'):
                usage_info["model"] = response.model

        except Exception as e:
            print(f"Usage info extraction error: {e}")

        return usage_info


# ================================================
# 設定と環境管理（型安全版）
# ================================================

class EnvironmentManager:
    """環境設定管理クラス"""

    REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]
    OPTIONAL_ENV_VARS = [
        "GITHUB_TOKEN",
        "SLACK_BOT_TOKEN",
        "GOOGLE_DRIVE_TOKEN",
        "DB_API_KEY"
    ]

    @classmethod
    def check_environment(cls) -> Dict[str, Any]:
        """環境設定をチェック"""
        status = {
            "required": {},
            "optional": {},
            "issues"  : [],
            "ready"   : True
        }

        # 必須環境変数のチェック
        for var in cls.REQUIRED_ENV_VARS:
            is_set = bool(os.getenv(var))
            status["required"][var] = is_set
            if not is_set:
                status["issues"].append(f"Required: {var} not set")
                status["ready"] = False

        # オプション環境変数のチェック
        for var in cls.OPTIONAL_ENV_VARS:
            is_set = bool(os.getenv(var))
            status["optional"][var] = is_set
            if not is_set:
                status["issues"].append(f"Optional: {var} not set")

        return status

    @classmethod
    def print_environment_status(cls):
        """環境設定状況を表示"""
        status = cls.check_environment()

        print("=== Environment Status ===")
        print(f"Overall Ready: {'✅' if status['ready'] else '❌'}")
        print()

        print("Required Environment Variables:")
        for var, is_set in status["required"].items():
            icon = "✅" if is_set else "❌"
            print(f"  {icon} {var}")

        print("\nOptional Environment Variables:")
        for var, is_set in status["optional"].items():
            icon = "✅" if is_set else "⚠️"
            print(f"  {icon} {var}")

        if status["issues"]:
            print(f"\nIssues ({len(status['issues'])}):")
            for issue in status["issues"]:
                print(f"  - {issue}")

        print(f"\nType Support: {'✅' if TYPES_AVAILABLE else '⚠️'}")
        print(f"Pydantic Support: {'✅' if PYDANTIC_AVAILABLE else '⚠️'}")


# ================================================
# メイン実行関数
# ================================================

async def main():
    """メイン実行関数"""
    print("=== OpenAI API + MCP 完全型対応版 ===\n")

    # 環境チェック
    EnvironmentManager.print_environment_status()

    # デモマネージャーの初期化
    manager = TypedDemoManager()

    print(f"\n=== Available Demos ({len(manager.list_demos())}) ===")
    for demo_name in manager.list_demos():
        print(f"  - {demo_name}")

    # 環境が整っている場合のテスト実行
    env_status = EnvironmentManager.check_environment()
    if env_status["ready"]:
        print("\n=== Running Test Demo ===")
        try:
            # 非同期デモのテスト
            # result = await manager.run_demo_async("github_mcp")
            # processor = TypedResponseProcessor()
            # content = processor.extract_content(result)
            # tool_calls = processor.extract_tool_calls(result)
            # usage = processor.get_usage_info(result)

            # print(f"Response Content: {content[:200]}...")
            # print(f"Tool Calls: {len(tool_calls)}")
            # print(f"Usage: {usage}")

            print("Test demo ready to run (uncomment to execute)")

        except Exception as e:
            print(f"Test demo error: {e}")
    else:
        print("\n⚠️ Environment not ready for execution")


def run_sync_demo(demo_name: str = "github_mcp"):
    """同期デモの実行"""
    print(f"=== Running Sync Demo: {demo_name} ===")

    manager = TypedDemoManager()
    processor = TypedResponseProcessor()

    try:
        result = manager.run_demo_sync(demo_name)
        content = processor.extract_content(result)
        tool_calls = processor.extract_tool_calls(result)
        usage = processor.get_usage_info(result)

        print(f"✅ Demo completed successfully")
        print(f"Content length: {len(content)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Usage: {usage}")

        return result

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


async def run_async_demo(demo_name: str = "multi_mcp"):
    """非同期デモの実行"""
    print(f"=== Running Async Demo: {demo_name} ===")

    manager = TypedDemoManager()
    processor = TypedResponseProcessor()

    try:
        result = await manager.run_demo_async(demo_name)
        content = processor.extract_content(result)
        tool_calls = processor.extract_tool_calls(result)
        usage = processor.get_usage_info(result)

        print(f"✅ Async demo completed successfully")
        print(f"Content length: {len(content)}")
        print(f"Tool calls: {len(tool_calls)}")
        print(f"Usage: {usage}")

        return result

    except Exception as e:
        print(f"❌ Async demo failed: {e}")
        return None


async def run_streaming_demo():
    """ストリーミングデモの実行"""
    print("=== Running Streaming Demo ===")

    manager = TypedDemoManager()

    try:
        result = await manager.run_streaming_demo("streaming")
        print(f"\n✅ Streaming completed. Total length: {len(result)}")
        return result

    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")
        return None


# ================================================
# カスタムデモの追加例
# ================================================

class CustomTypedDemo(BaseTypedDemo):
    """カスタムデモの実装例"""

    def __init__(self):
        super().__init__("Custom Typed Demo")

    def create_tools(self) -> List[Union[ResponseTool, Dict[str, Any]]]:
        """カスタムツールを作成"""
        # 例：独自のMCPサーバー設定
        custom_config = MCPServerConfig(
            server_url="https://my-custom-mcp.example.com",
            server_label="custom",
            headers={"Authorization": f"Bearer {os.getenv('CUSTOM_API_KEY')}"},
            allowed_tools=["custom_function_1", "custom_function_2"],
            require_approval="never"
        )

        return [
            self.tool_factory.create_mcp_tool(custom_config),
            self.tool_factory.create_code_interpreter_tool()
        ]

    def get_input_message(self) -> str:
        return "Execute custom analysis using our proprietary MCP server and code interpreter."


# ================================================
# 実行部分
# ================================================

if __name__ == "__main__":
    # 基本的な環境チェックとデモリスト表示
    asyncio.run(main())

    print("\n=== Usage Examples ===")
    print("Sync demo:")
    print("  result = run_sync_demo('github_mcp')")
    print()
    print("Async demo:")
    print("  result = await run_async_demo('multi_mcp')")
    print()
    print("Streaming demo:")
    print("  result = await run_streaming_demo()")
    print()
    print("Custom demo:")
    print("  manager = TypedDemoManager()")
    print("  manager.add_demo('custom', CustomTypedDemo())")
    print("  result = manager.run_demo_sync('custom')")

"""
=== 完全型対応版の特徴 ===

1. 完全な型安全性:
   - すべてのツールで適切な型付きオブジェクトを使用
   - OpenAI Python SDKの正式な型定義を活用
   - Pydanticによるデータ検証

2. 包括的なツール対応:
   - MCPTool: MCP サーバー用
   - CodeInterpreterTool: コード実行用
   - WebSearchPreviewTool: Web検索用
   - ImageGenerationTool: 画像生成用

3. エラーハンドリング:
   - 型が利用できない場合のフォールバック実装
   - 環境設定の自動チェック
   - 詳細なエラー情報

4. 柔軟性:
   - 同期・非同期両対応
   - ストリーミング対応
   - カスタムデモの簡単追加

5. 実用性:
   - レスポンス処理ユーティリティ
   - 使用量情報の抽出
   - デバッグ支援機能

この実装により、型エラーが完全に解決され、
最新のOpenAI Python SDKで安全かつ効率的に
MCPサーバーを利用できます。
"""