# 概要書

## 処理の概要
`a30_30_rag_search.py` は Streamlit を利用した RAG (Retrieval Augmented Generation) 検索アプリである。環境変数 `OPENAI_API_KEY` からキーを読み取り、OpenAI Responses API と `file_search` ツールで Vector Store を検索する。必要に応じて Agent SDK を用いたセッション管理も可能。検索結果は引用ファイル情報とともに表示し、履歴や検索オプションを管理できる。

## main の処理の流れ
1. ページ設定とセッション状態初期化
2. `get_rag_manager` で `ModernRAGManager` インスタンス取得
3. ヘッダーおよび API 利用状況の表示
4. サイドバーで Vector Store・言語・検索オプション等を設定し、テスト質問を表示
5. メインコンテンツ左側で質問入力フォームを表示し、送信時 `rag_manager.search` で検索
6. 右側に検索結果やガイド等を表示。履歴管理やフッター出力
7. スクリプトが直接実行された場合に `main()` が起動

---

# 関数一覧

| 関数・メソッド名 | 処理概要 |
| --- | --- |
| `ModernRAGManager.__init__` | Agent SDK 用セッション管理辞書を初期化 |
| `ModernRAGManager.search_with_responses_api` | Responses API + `file_search` による検索処理 |
| `ModernRAGManager.search_with_agent_sdk` | Agent SDK による簡易検索。失敗時は Responses API にフォールバック |
| `ModernRAGManager.search` | 設定に応じて上記メソッドを呼び分ける統合検索 |
| `ModernRAGManager._extract_response_text` | Responses API の返却データから回答テキストを抽出 |
| `ModernRAGManager._extract_citations` | レスポンスから引用ファイル情報を取得 |
| `ModernRAGManager._extract_tool_calls` | `file_search` 呼び出し情報を取得 |
| `get_rag_manager` | `ModernRAGManager` インスタンスを取得 (Streamlit キャッシュ) |
| `initialize_session_state` | 検索履歴などのセッション初期値を登録 |
| `display_search_history` | 検索履歴を表示し再実行・詳細表示ボタンを提供 |
| `get_selected_store_index` | 選択 Vector Store 名からインデックスを取得 |
| `display_test_questions` | Vector Store/言語に応じたテスト質問ボタンを表示 |
| `display_system_info` | 利用可能機能や設定値などをサイドバーで表示 |
| `display_search_options` | 検索オプション (結果数・詳細表示等) を設定 |
| `display_search_results` | 検索結果本文、引用ファイル、メタ情報を表示 |
| `main` | UI 構築と検索実行を行うエントリーポイント |

---

# 関数の詳細設計

## `ModernRAGManager.__init__`
- **処理概要**
  Agent SDK 用セッションを保持する辞書を初期化。
- **処理の流れ**
  1. `self.agent_sessions` を空辞書で生成。
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: セッション辞書生成
  - **OUTPUT**: 初期状態のインスタンス

## `ModernRAGManager.search_with_responses_api`
- **処理概要**
  Responses API と `file_search` を用いて Vector Store 検索を実行し、結果とメタデータを返す。
- **処理の流れ**
  1. Vector Store ID を取得
  2. `file_search` 設定生成 (結果数やフィルタを適用)
  3. Responses API 呼び出し
  4. `_extract_response_text` で回答を抽出
  5. `_extract_citations` で引用情報取得
  6. メタデータを構築 (使用統計含む)
  7. 正常時は結果とメタデータ、例外時はエラーメタデータを返す
- **IPO**
  - **INPUT**: `query`, `store_name`, `max_results` など
  - **PROCESS**: file_search 設定 → API 呼び出し → テキスト/引用抽出 → メタ生成
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager.search_with_agent_sdk`
- **処理概要**
  Agent SDK を用いた簡易検索。必要に応じて Responses API へフォールバック。
- **処理の流れ**
  1. Agent SDK の利用可否を確認。不可能なら Responses API を利用
  2. セッション取得/生成
  3. 簡易 Agent を作成し Runner 実行
  4. 出力テキストとメタデータを生成
  5. 例外時は Responses API で検索しなおす
- **IPO**
  - **INPUT**: `query`, `store_name`
  - **PROCESS**: セッション確保 → Agent 実行 → メタデータ生成 → 例外時フォールバック
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager.search`
- **処理概要**
  `use_agent_sdk` の設定に基づき検索メソッドを選択。
- **処理の流れ**
  1. Agent SDK 利用設定と可否を判断
  2. 条件に応じ `search_with_agent_sdk` または `search_with_responses_api` を実行
- **IPO**
  - **INPUT**: `query`, `store_name`, `use_agent_sdk`, 他オプション
  - **PROCESS**: 条件判定 → それぞれの検索実行
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager._extract_response_text`
- **処理概要**
  Responses API のレスポンスから回答テキストを取り出す。
- **処理の流れ**
  1. `output_text` 属性があればそのまま返す
  2. ない場合、`output` 配列から `message` → `output_text` を探索
  3. 見つからなければエラーメッセージを返す
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: 属性の確認・走査
  - **OUTPUT**: 取得したテキスト

## `ModernRAGManager._extract_citations`
- **処理概要**
  引用ファイル情報をレスポンスから抽出。
- **処理の流れ**
  1. `response.output` を走査し `file_citation` アノテーションを収集
  2. 例外時はログ出力のみ
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: 出力内のアノテーション走査
  - **OUTPUT**: 引用情報リスト (`List[Dict[str, Any]]`)

## `ModernRAGManager._extract_tool_calls`
- **処理概要**
  `file_search` ツール呼び出し結果を抽出する。
- **処理の流れ**
  1. `response.output` を走査して `file_search_call` を収集
  2. 例外時はログ出力
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: ツール呼び出し情報抽出
  - **OUTPUT**: ツール呼び出し情報リスト

## `get_rag_manager`
- **処理概要**
  `ModernRAGManager` インスタンスを生成・キャッシュして返す。
- **処理の流れ**
  1. `ModernRAGManager` を作成し Streamlit キャッシュに保存して返す
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: インスタンス生成・キャッシュ
  - **OUTPUT**: `ModernRAGManager` インスタンス

## `initialize_session_state`
- **処理概要**
  Streamlit のセッション状態に初期値を設定。
- **処理の流れ**
  1. `search_history` などのキーが存在しない場合にデフォルト値を設定
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: `st.session_state` の初期化
  - **OUTPUT**: なし（状態更新）

## `display_search_history`
- **処理概要**
  検索履歴をエクスパンダーで表示し再実行・詳細ボタンを提供。
- **処理の流れ**
  1. 履歴がない場合メッセージ表示
  2. 最新 10 件を一覧。再実行／詳細表示ボタンを設置
- **IPO**
  - **INPUT**: `st.session_state.search_history`
  - **PROCESS**: 履歴出力・ボタン処理
  - **OUTPUT**: 画面更新のみ

## `get_selected_store_index`
- **処理概要**
  選択 Vector Store 名からインデックスを取得。
- **処理の流れ**
  1. `VECTOR_STORE_LIST.index()` を用いてインデックスを返す。見つからなければ 0
- **IPO**
  - **INPUT**: `selected_store`
  - **PROCESS**: インデックス探索
  - **OUTPUT**: `int` インデックス

## `display_test_questions`
- **処理概要**
  Vector Store と言語に応じたテスト質問をボタン表示。
- **処理の流れ**
  1. 選択中の設定から質問リストを取得
  2. 注意書きを表示し、各質問をボタンとして配置
- **IPO**
  - **INPUT**: `st.session_state`
  - **PROCESS**: リスト取得 → ボタン表示
  - **OUTPUT**: ボタン押下時はセッション更新

## `display_system_info`
- **処理概要**
  利用可能機能や設定情報をサイドバー内で表示。
- **処理の流れ**
  1. OpenAI SDK / Agent SDK の可否、API キー設定方法等を列挙
  2. Vector Store 情報や現在の設定を表示
- **IPO**
  - **INPUT**: `st.session_state`
  - **PROCESS**: 文字列組み立て → 画面出力
  - **OUTPUT**: 画面更新

## `display_search_options`
- **処理概要**
  検索結果数や詳細表示、引用表示、Agent SDK 使用などを設定する UI。
- **処理の流れ**
  1. スライダー・チェックボックスで値入力
  2. 入力された値を `st.session_state` に保存
- **IPO**
  - **INPUT**: 既存のセッション値
  - **PROCESS**: UI 生成 → 値を保存
  - **OUTPUT**: 状態更新

## `display_search_results`
- **処理概要**
  検索結果と引用ファイル・メタ情報を表示。
- **処理の流れ**
  1. 回答テキストを出力
  2. 引用表示が有効なら引用ファイル一覧を列挙
  3. メタデータを表形式で表示、詳細はエクスパンダーに格納
- **IPO**
  - **INPUT**: `response_text`, `metadata`
  - **PROCESS**: Markdown 出力による情報表示
  - **OUTPUT**: 画面更新

## `main`
- **処理概要**
  Streamlit アプリを構築し検索を実行するエントリーポイント。
- **処理の流れ**
  1. ページ設定 → `initialize_session_state` → `get_rag_manager`
  2. ヘッダー・API 状況表示
  3. サイドバーで各種設定・テスト質問・システム情報を表示
  4. メインカラムで質問フォームを表示し、送信時に検索実行 (`rag_manager.search`)
  5. 結果表示後、履歴へ保存
  6. クエリが未入力の場合は機能説明やトラブルシューティング情報を表示
  7. 検索履歴とフッターを表示して終了
- **IPO**
  - **INPUT**: ユーザーからの質問や設定値
  - **PROCESS**: UI 表示 → 入力取得 → `rag_manager` による検索 → 結果表示・履歴管理
  - **OUTPUT**: 検索結果画面

