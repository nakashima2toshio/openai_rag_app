# 概要書

## 処理の概要
`a30_30_rag_search.py` は Streamlit を用いた RAG（Retrieval Augmented Generation）検索アプリケーションである。環境変数 `OPENAI_API_KEY` から OpenAI API キーを取得し、OpenAI Responses API の `file_search` ツールを利用して Vector Store から検索を行う。必要に応じて Agent SDK を介したセッション管理も可能。検索結果は引用ファイル情報とともに表示し、履歴や検索オプションを管理できる。

## main の処理の流れ
1. ページ設定とセッション状態初期化
2. `get_rag_manager` で `ModernRAGManager` インスタンスを取得
3. ヘッダーと API 利用状況を表示
4. サイドバーで Vector Store／言語／検索オプションを設定し、テスト質問を表示
5. メインコンテンツ左側で質問入力フォームを表示し、検索実行時 `rag_manager.search` を呼び出して結果取得
6. 右側に検索結果やガイドを表示。履歴管理やフッター表示も行う
7. スクリプトが直接実行された場合 `main()` を起動

---

# 関数一覧

| 関数名 / メソッド名 | 処理概要 |
| --- | --- |
| `ModernRAGManager.__init__` | Agent SDK 用セッション管理辞書を初期化 |
| `ModernRAGManager.search_with_responses_api` | Responses API と `file_search` ツールを用いた検索処理 |
| `ModernRAGManager.search_with_agent_sdk` | Agent SDK を利用した検索（RAG 機能は Responses API に委譲） |
| `ModernRAGManager.search` | 上記メソッドを呼び分ける統合検索 |
| `ModernRAGManager._extract_response_text` | レスポンスから回答テキストを抽出 |
| `ModernRAGManager._extract_citations` | 引用ファイル情報を取得 |
| `ModernRAGManager._extract_tool_calls` | `file_search` 呼び出し情報を取得 |
| `get_rag_manager` | `ModernRAGManager` を Streamlit キャッシュから取得 |
| `initialize_session_state` | セッションに検索履歴や設定の初期値を登録 |
| `display_search_history` | 検索履歴をエクスパンダーで表示 |
| `get_selected_store_index` | 選択 Vector Store 名からインデックスを取得 |
| `display_test_questions` | Vector Store と言語に応じたテスト質問をボタン表示 |
| `display_system_info` | 機能や設定をサイドバーに表示 |
| `display_search_options` | 検索オプションの入力を受け付ける |
| `display_search_results` | 検索結果・引用・メタ情報を表示 |
| `main` | アプリケーションのエントリーポイント |

---

# 関数の詳細設計

## `ModernRAGManager.__init__`
- **処理概要**
  Agent SDK 用セッション管理辞書を初期化する。
- **処理の流れ**
  1. `self.agent_sessions` を空辞書として生成する。
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: セッション辞書生成
  - **OUTPUT**: 初期化済みインスタンス

## `ModernRAGManager.search_with_responses_api`
- **処理概要**
  `file_search` を利用して Responses API から検索結果とメタデータを取得。
- **処理の流れ**
  1. Vector Store ID 取得
  2. `file_search` 設定を作成しオプションを追加
  3. Responses API を呼び出す
  4. `_extract_response_text` と `_extract_citations` で情報抽出
  5. 使用統計があればメタデータに含める
  6. 成功時は結果とメタデータ、例外時はエラー情報を返す
- **IPO**
  - **INPUT**: `query`, `store_name`, `max_results` など
  - **PROCESS**: file_search 設定 → API 呼び出し → テキスト・引用抽出 → メタ生成
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager.search_with_agent_sdk`
- **処理概要**
  Agent SDK を利用した検索。失敗時は Responses API へフォールバック。
- **処理の流れ**
  1. Agent SDK の可否確認、不可なら Responses API に切り替え
  2. セッションを取得／生成
  3. 簡易 Agent を作成し Runner を実行
  4. 出力テキストとメタデータをまとめる
  5. 例外時に Responses API で検索しなおす
- **IPO**
  - **INPUT**: `query`, `store_name`
  - **PROCESS**: セッション取得 → Agent 実行 → メタ生成 → フォールバック
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager.search`
- **処理概要**
  `use_agent_sdk` 設定に応じて検索メソッドを選択。
- **処理の流れ**
  1. Agent SDK 使用可否を判定
  2. 該当メソッドを呼び出し結果を返す
- **IPO**
  - **INPUT**: `query`, `store_name`, `use_agent_sdk`, 他オプション
  - **PROCESS**: 条件判断 → 検索実行
  - **OUTPUT**: `(response_text, metadata)` タプル

## `ModernRAGManager._extract_response_text`
- **処理概要**
  Responses API のレスポンスから回答テキストを抽出。
- **処理の流れ**
  1. `output_text` 属性があれば返す
  2. `output` 配列を走査し `message` の `output_text` を探す
  3. 見つからなければエラーメッセージを返す
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: 属性チェック → 配列走査
  - **OUTPUT**: テキスト文字列

## `ModernRAGManager._extract_citations`
- **処理概要**
  レスポンスから引用ファイル情報を取得。
- **処理の流れ**
  1. `response.output` 内のアノテーションを探索
  2. `file_citation` を収集しリスト化する
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: アノテーション走査
  - **OUTPUT**: `List[Dict[str, Any]]`

## `ModernRAGManager._extract_tool_calls`
- **処理概要**
  `file_search` 呼び出し情報を抽出する。
- **処理の流れ**
  1. `response.output` を巡回し `file_search_call` を取得
  2. 情報を辞書形式でリストに保存
- **IPO**
  - **INPUT**: `response`
  - **PROCESS**: 出力要素から情報抽出
  - **OUTPUT**: `List[Dict[str, Any]]`

## `get_rag_manager`
- **処理概要**
  `ModernRAGManager` インスタンスを Streamlit のキャッシュから取得。
- **処理の流れ**
  1. `ModernRAGManager` を生成・返却
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: インスタンス生成（キャッシュ利用）
  - **OUTPUT**: `ModernRAGManager` インスタンス

## `initialize_session_state`
- **処理概要**
  セッション状態に検索履歴や各種設定の初期値を登録。
- **処理の流れ**
  1. 履歴・質問・Vector Store 等のキーを確認しデフォルト値を設定
- **IPO**
  - **INPUT**: なし
  - **PROCESS**: `st.session_state` のキー確認・初期化
  - **OUTPUT**: なし

## `display_search_history`
- **処理概要**
  検索履歴を表示し再実行や詳細ボタンを提供。
- **処理の流れ**
  1. 履歴が無い場合メッセージ表示
  2. 最新 10 件をエクスパンダーで表示しボタンを設置
- **IPO**
  - **INPUT**: `st.session_state.search_history`
  - **PROCESS**: 履歴一覧生成・ボタン処理
  - **OUTPUT**: 画面表示

## `get_selected_store_index`
- **処理概要**
  選択中 Vector Store 名からインデックスを算出。
- **処理の流れ**
  1. `VECTOR_STORE_LIST.index()` を使用し、無ければ 0 を返す
- **IPO**
  - **INPUT**: `selected_store`
  - **PROCESS**: インデックス検索
  - **OUTPUT**: `int`

## `display_test_questions`
- **処理概要**
  Vector Store と言語に応じたテスト質問をボタンで表示。
- **処理の流れ**
  1. 選択情報から質問リストを取得
  2. 注意書きを表示しボタンを配置
- **IPO**
  - **INPUT**: `st.session_state`
  - **PROCESS**: 質問取得 → ボタン生成
  - **OUTPUT**: ボタン押下によりセッション更新

## `display_system_info`
- **処理概要**
  利用可能機能や現在の設定をサイドバーで表示。
- **処理の流れ**
  1. OpenAI SDK 等の可否を表示
  2. API キー設定方法や Vector Store 一覧を出力
- **IPO**
  - **INPUT**: `st.session_state`
  - **PROCESS**: 情報整形 → 表示
  - **OUTPUT**: 画面出力

## `display_search_options`
- **処理概要**
  検索オプション（結果数・詳細表示・引用表示・Agent 使用）の入力 UI を提供。
- **処理の流れ**
  1. スライダーやチェックボックスで値を取得
  2. `st.session_state.search_options` に保存
- **IPO**
  - **INPUT**: 既存のセッション値
  - **PROCESS**: UI 生成 → セッション保存
  - **OUTPUT**: なし

## `display_search_results`
- **処理概要**
  検索結果本文と引用ファイル、メタ情報を表示。
- **処理の流れ**
  1. 回答テキストを表示
  2. 引用ファイルをリスト表示
  3. メタデータを表形式で表示し詳細はエクスパンダーに格納
- **IPO**
  - **INPUT**: `response_text`, `metadata`
  - **PROCESS**: Markdown による情報表示
  - **OUTPUT**: 画面更新

## `main`
- **処理概要**
  Streamlit アプリのエントリーポイントとして UI 構築と検索処理を実行。
- **処理の流れ**
  1. ページ設定・セッション初期化
  2. RAG マネージャ取得
  3. ヘッダーやサイドバーを設定
  4. 質問フォームから入力を受け `rag_manager.search` を実行
  5. 結果表示・履歴保存。未入力時は機能説明を表示
  6. 検索履歴とフッターを出力して終了
- **IPO**
  - **INPUT**: ユーザー質問、サイドバー設定
  - **PROCESS**: UI 表示 → 入力取得 → 検索実行 → 結果表示・履歴管理
  - **OUTPUT**: 検索結果画面


