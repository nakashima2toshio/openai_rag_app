# Streamlit UIヘルパーライブラリ詳細設計書

## 概要書

### 処理の概要
本プログラムは、Streamlitを使用したOpenAI APIデモアプリケーションのUI機能を簡素化するためのヘルパーライブラリです。`helper_api.py`と連携し、Streamlit固有のUI コンポーネント管理、セッション状態管理、エラーハンドリング、パフォーマンス監視、情報パネル表示などの機能を提供します。

**主要機能：**
1. **安全なStreamlit表示**: JSON表示のエラー耐性機能
2. **UI用デコレータ**: Streamlit環境でのエラーハンドリング、タイマー、キャッシュ
3. **セッション状態管理**: Streamlitセッション状態の統合管理とユーザー設定保存
4. **メッセージ管理（UI版）**: Streamlitセッション状態を使用したメッセージ履歴管理
5. **UI ヘルパー**: ページ初期化、モデル選択、フォーム作成、メトリクス表示等
6. **レスポンス処理（UI版）**: APIレスポンスのStreamlit表示とダウンロード機能
7. **デモ基底クラス**: 共通のデモアプリケーション開発フレームワーク
8. **情報パネル管理**: 左サイドバーの情報パネル統合管理
9. **後方互換性**: 既存コードとの互換性維持

### mainの処理の流れ
**注意：本ライブラリにはmain関数は存在しません。**
これはStreamlitアプリケーション内でインポートして使用するライブラリとして設計されています。

**典型的なStreamlitアプリでの使用フロー：**
1. ライブラリのインポート
2. `UIHelper.init_page()`でページ初期化
3. `SessionStateManager.init_session_state()`でセッション状態初期化
4. `UIHelper.select_model()`でモデル選択UI表示
5. `UIHelper.create_input_form()`でユーザー入力フォーム作成
6. `DemoBase`継承クラスでデモアプリケーション開発
7. `ResponseProcessorUI.display_response()`でAPI結果表示
8. 情報パネル（モデル情報、コスト計算、パフォーマンス等）表示

---

## クラス・関数一覧

### クラス一覧

| クラス名 | 処理概要 |
|----------|----------|
| `SessionStateManager` | Streamlitセッション状態の統合管理。初期化、ユーザー設定保存、キャッシュ管理、パフォーマンス記録 |
| `MessageManagerUI` | Streamlitセッション状態を使用したメッセージ履歴管理。`MessageManager`のUI拡張版 |
| `UIHelper` | Streamlit UI用統合ヘルパー。ページ初期化、モデル選択、フォーム作成、表示機能等 |
| `ResponseProcessorUI` | APIレスポンスのStreamlit表示処理。`ResponseProcessor`のUI拡張版 |
| `DemoBase` | デモアプリケーション開発用の抽象基底クラス。共通UI設定とAPI呼び出し機能 |
| `InfoPanelManager` | 左サイドバーの情報パネル統合管理。モデル情報、コスト計算、パフォーマンス表示等 |

### 関数一覧

| 関数名 | 処理概要 |
|--------|----------|
| `safe_streamlit_json()` | Streamlit用安全JSON表示。エラー耐性とフォールバック機能 |
| `init_page()` | 後方互換性のためのページ初期化関数 |
| `init_messages()` | 後方互換性のためのメッセージ初期化関数 |
| `select_model()` | 後方互換性のためのモデル選択関数 |
| `get_default_messages()` | 後方互換性のためのデフォルトメッセージ取得関数 |
| `extract_text_from_response()` | 後方互換性のためのレスポンステキスト抽出関数 |
| `append_user_message()` | 後方互換性のためのユーザーメッセージ追加関数（画像対応） |

### デコレータ一覧

| デコレータ名 | 処理概要 |
|------------|----------|
| `@error_handler_ui` | Streamlit用エラーハンドリング。エラー表示、デバッグ情報、ログ出力 |
| `@timer_ui` | Streamlit用実行時間計測。パフォーマンス記録とセッション状態保存 |
| `@cache_result_ui(ttl)` | Streamlitセッション状態を使用した結果キャッシュ |

---

## クラスの詳細設計書

### 1. SessionStateManager

#### 処理概要
Streamlitのセッション状態（`st.session_state`）を統合管理する静的クラス。アプリケーション全体のセッション状態初期化、ユーザー設定の保存・取得、UIキャッシュ管理、パフォーマンスメトリクス記録機能を提供します。

#### 処理の流れ
1. セッション状態の初期化（`initialized`フラグ確認）
2. 必要なセッション変数の作成（`ui_cache`, `performance_metrics`, `user_preferences`）
3. ユーザー設定の永続化管理
4. キャッシュの統合管理（メモリとセッション状態の両方）
5. パフォーマンスメトリクスの収集・提供

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - セッション状態キー（文字列）
  - ユーザー設定値（Any型）
  - パフォーマンスデータ

- **PROCESS：**
  - Streamlitセッション状態の初期化・管理
  - ユーザー設定の永続化
  - キャッシュサイズ制限・クリア処理
  - パフォーマンスデータ収集・集計

- **OUTPUT：**
  - ユーザー設定値
  - パフォーマンスメトリクスリスト
  - セッション状態の統合管理機能

---

### 2. MessageManagerUI

#### 処理概要
`MessageManager`をStreamlit環境に拡張したクラス。Streamlitのセッション状態を使用してメッセージ履歴を永続化し、UI設定に基づくメッセージ数制限、インポート/エクスポート機能を提供します。

#### Streamlit拡張機能
- セッション状態での履歴永続化
- UI設定ベースのメッセージ表示制限
- セッションキーのカスタマイズ対応
- UI用JSON出力機能

#### 処理の流れ
1. セッションキーベースのメッセージ履歴初期化
2. Streamlitセッション状態への履歴保存
3. UI設定（`ui.message_display_limit`）に基づく制限適用
4. 開発者メッセージ保護（制限時も保持）
5. UI用のエクスポート機能（JSON文字列化）

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - セッションキー（デフォルト："message_history"）
  - ロール（user/assistant/system/developer）
  - メッセージ内容
  - インポートデータ（辞書）

- **PROCESS：**
  - Streamlitセッション状態でのメッセージ管理
  - UI設定ベースの制限適用
  - 開発者メッセージ保護処理
  - 安全なJSON文字列化

- **OUTPUT：**
  - セッション状態保存されたメッセージ履歴
  - UI用JSON文字列（エクスポート）
  - 永続化されたメッセージ履歴

---

### 3. UIHelper

#### 処理概要
Streamlit UI開発のための統合ヘルパークラス。ページ初期化、モデル選択UI、入力フォーム作成、メッセージ表示、トークン情報表示、各種UI コンポーネント作成機能を提供する大規模なユーティリティクラスです。

#### 主要メソッド群
- **ページ管理**: `init_page()`, `_show_debug_info()`
- **モデル選択**: `select_model()` （カテゴリ対応）
- **フォーム作成**: `create_input_form()` （複数入力タイプ対応）
- **表示機能**: `display_messages()`, `show_token_info()`, `show_metrics()`
- **UI構築**: `create_tabs()`, `create_columns()`, `create_download_button()`
- **設定管理**: `show_settings_panel()`, `show_performance_panel()`

#### 処理の流れ
1. Streamlitページ設定（`st.set_page_config()`）の安全な実行
2. セッション状態の自動初期化
3. 設定ファイルベースのUI設定適用
4. デバッグ情報の条件付き表示
5. ユーザー設定の永続化

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - ページタイトル、レイアウト設定
  - モデル選択カテゴリ
  - フォーム設定（入力タイプ、ラベル等）
  - 表示データ（メッセージ、メトリクス等）
  - 設定ファイルの各種設定

- **PROCESS：**
  - Streamlitページ設定の安全な適用
  - 設定ファイルベースのUI構築
  - セッション状態とユーザー設定の統合管理
  - 安全なJSON処理とエラーハンドリング
  - カテゴリ別モデルフィルタリング

- **OUTPUT：**
  - 設定済みStreamlitページ
  - 選択されたモデル名
  - ユーザー入力データと送信状態
  - 構築されたUIコンポーネント
  - ダウンロード可能なデータファイル

---

### 4. ResponseProcessorUI

#### 処理概要
`ResponseProcessor`をStreamlit環境に拡張したクラス。APIレスポンスのStreamlit表示、詳細情報パネル、コスト計算表示、安全なJSON表示、ダウンロードボタン付きレスポンス保存機能を提供します。

#### Streamlit拡張機能
- レスポンステキストのStreamlit表示（コピーボタン付き）
- 詳細情報の折りたたみ表示
- トークン使用量の視覚的表示
- コスト計算とメトリクス表示
- 安全なJSON表示とダウンロード機能

#### 処理の流れ
1. レスポンステキストの抽出と表示
2. コピーボタン付きテキスト表示
3. 詳細情報パネルの作成（使用状況、レスポンス情報）
4. トークン使用量の可視化（メトリクス表示）
5. コスト計算の表示
6. Raw JSON の安全な表示
7. ダウンロードボタンの作成

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - OpenAI Responseオブジェクト
  - 表示オプション（詳細表示、Raw表示）

- **PROCESS：**
  - レスポンステキスト抽出と整形
  - トークン使用量の視覚化
  - コスト計算処理
  - 安全なJSON表示処理
  - エラーハンドリングとフォールバック

- **OUTPUT：**
  - Streamlit表示されたレスポンス
  - メトリクス表示（トークン数、コスト）
  - ダウンロード可能なJSONファイル
  - エラー表示（該当する場合）

---

### 5. DemoBase

#### 処理概要
デモアプリケーション開発用の抽象基底クラス。共通のUI設定、メッセージ管理、API呼び出し機能を提供し、具体的なデモクラスの実装を簡素化します。抽象メソッド`run()`をサブクラスで実装することで、統一されたデモアプリケーションを構築できます。

#### 共通機能
- デモ名ベースのキー管理
- MessageManagerUIの自動初期化
- 共通UI設定（モデル選択、設定パネル、履歴クリア）
- 統一されたAPI呼び出し処理
- エラーハンドリングとタイマー機能

#### 処理の流れ
1. デモ名とタイトルの設定
2. キープレフィックスの生成（安全な文字列化）
3. MessageManagerUIの初期化（セッションキー付き）
4. セッション状態の初期化
5. 共通UI設定の提供
6. API呼び出しの統一処理

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - デモ名、タイトル
  - APIパラメータ
  - メッセージ内容

- **PROCESS：**
  - キー管理とセッション状態初期化
  - 共通UI設定の構築
  - メッセージ管理の統合
  - API呼び出しの標準化

- **OUTPUT：**
  - 設定済みデモ環境
  - API レスポンス
  - 統一されたUI表示

---

### 6. InfoPanelManager

#### 処理概要
左サイドバーの情報パネルを統合管理する静的クラス。モデル情報、セッション情報、コスト計算、パフォーマンス情報、デバッグ情報、設定パネルを提供し、アプリケーションの状態を視覚化します。

#### パネル種類
- **モデル情報**: 制限、料金、特性表示
- **セッション情報**: アクティブセッション、メッセージ統計
- **コスト計算**: 料金シミュレーター、月間推定
- **パフォーマンス**: 実行時間統計、関数呼び出し情報
- **デバッグ**: 設定値、ログレベル、キャッシュ状態
- **設定**: ユーザー設定の管理

#### 処理の流れ
1. 各パネルの展開可能エリア作成
2. 設定ファイルからの情報取得
3. セッション状態からの統計取得
4. リアルタイム情報の表示
5. ユーザー操作への応答処理

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - 選択されたモデル名
  - セッション状態データ
  - 設定ファイル情報
  - パフォーマンスメトリクス

- **PROCESS：**
  - 情報の集計・計算処理
  - UI コンポーネントの生成
  - ユーザー操作の処理
  - リアルタイム更新

- **OUTPUT：**
  - サイドバー情報パネル
  - ユーザー設定の更新
  - システム状態の表示

---

## 関数の詳細設計書

### 1. safe_streamlit_json()

#### 処理概要
Streamlitの`st.json()`機能を安全に実行する関数。OpenAI APIオブジェクトなど、標準のJSONシリアライゼーションで問題が生じるデータを安全に表示します。

#### 処理の流れ
1. 直接`st.json()`での表示を試行
2. 失敗時：カスタムシリアライザー（`safe_json_dumps()`）を使用してリトライ
3. 再度失敗時：フォールバック処理でコードブロック表示
4. エラーメッセージの表示

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：** Any型データ、展開フラグ
- **PROCESS：** 段階的フォールバック処理、エラーハンドリング
- **OUTPUT：** Streamlit JSON表示またはコードブロック表示

---

### 2. 後方互換性関数群

#### init_page()
`UIHelper.init_page()`のラッパー関数。既存コードとの互換性を維持。

#### init_messages()
MessageManagerUIの初期化とクリアボタン付きメッセージ管理。

#### select_model()
`UIHelper.select_model()`のラッパー関数。デモ名ベースのキー生成。

#### get_default_messages()
MessageManagerUIを使用したデフォルトメッセージ取得。

#### extract_text_from_response()
`ResponseProcessor.extract_text()`のラッパー関数。

#### append_user_message()
画像URL対応のユーザーメッセージ追加。マルチモーダル対応。

---

## デコレータ詳細

### @error_handler_ui

#### 処理概要
Streamlit環境でのエラーハンドリング専用デコレータ。設定ファイルの多言語エラーメッセージを使用し、デバッグモード時には詳細な例外情報を表示します。

#### 処理の流れ
1. 関数実行の例外キャッチ
2. ログへのエラー記録
3. 設定ファイルからエラーメッセージ取得
4. `st.error()`でエラー表示
5. デバッグモード時：`st.exception()`で詳細表示

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：** デコレートされた関数
- **PROCESS：** 例外処理、ログ記録、UI表示
- **OUTPUT：** エラー表示、ログ出力、None（エラー時）

---

### @timer_ui

#### 処理概要
Streamlit環境での実行時間計測専用デコレータ。パフォーマンス監視が有効な場合、セッション状態にメトリクスを記録します。

#### 処理の流れ
1. 実行開始時刻の記録
2. 関数実行
3. 実行終了時刻の記録と実行時間計算
4. ログへの実行時間記録
5. パフォーマンス監視有効時：セッション状態への記録

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：** デコレートされた関数
- **PROCESS：** 時間計測、ログ記録、セッション状態更新
- **OUTPUT：** 関数実行結果、パフォーマンスデータ

---

### @cache_result_ui(ttl)

#### 処理概要
Streamlitセッション状態を使用した結果キャッシュデコレータ。TTL機能付きで、セッション単位でのキャッシュ管理を行います。

#### 処理の流れ
1. キャッシュ有効性の確認
2. ハッシュベースのキャッシュキー生成
3. セッション状態からキャッシュ検索
4. TTL検証とキャッシュヒット処理
5. キャッシュミス時：関数実行とキャッシュ保存
6. サイズ制限に基づく古いキャッシュ削除

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：** TTL設定、関数引数
- **PROCESS：** キャッシュ管理、TTL検証、サイズ制限
- **OUTPUT：** キャッシュされた結果または新規実行結果

---

## 設定ファイル連携

### 使用する設定セクション
- **ui**: ページ設定、テーマ、フォーム設定
- **models**: モデル選択、カテゴリ分類
- **model_pricing**: コスト計算
- **error_messages**: 多言語エラーメッセージ
- **cache**: キャッシュ設定
- **experimental**: デバッグ、パフォーマンス監視
- **logging**: ログ設定

### Streamlit固有設定
- **ui.page_title**: ページタイトル
- **ui.page_icon**: ページアイコン
- **ui.layout**: レイアウト（wide/centered）
- **ui.text_area_height**: テキストエリア高さ
- **ui.message_display_limit**: メッセージ表示制限

---

## 推奨使用パターン

### 基本的なStreamlitアプリ
```python
import streamlit as st
from helper_st import UIHelper, MessageManagerUI, ResponseProcessorUI

# ページ初期化
UIHelper.init_page("デモアプリ")

# モデル選択
model = UIHelper.select_model("demo")

# メッセージ管理
msg_manager = MessageManagerUI("demo_messages")

# 入力フォーム
user_input, submitted = UIHelper.create_input_form("demo_form", label="質問を入力")

if submitted and user_input:
    msg_manager.add_message("user", user_input)
    # API呼び出し・レスポンス表示
```

### デモアプリ開発
```python
from helper_st import DemoBase

class MyDemo(DemoBase):
    def __init__(self):
        super().__init__("my_demo", "私のデモ")

    def run(self):
        self.setup_ui()
        # デモ固有の処理

demo = MyDemo()
demo.run()
```

### 情報パネル表示
```python
from helper_st import InfoPanelManager

selected_model = UIHelper.select_model()
InfoPanelManager.show_model_info(selected_model)
InfoPanelManager.show_cost_info(selected_model)
InfoPanelManager.show_performance_info()
```

---

## 注意事項

### Streamlit固有の制約
- セッション状態は同一セッション内でのみ永続化
- `st.set_page_config()`は一度のみ実行可能
- ネストした`st.expander()`は推奨されない

### パフォーマンス考慮事項
- セッション状態のサイズ制限を考慮
- キャッシュの適切な使用でAPI呼び出し削減
- UI更新時の不要な再実行回避

### エラーハンドリング
- 多言語エラーメッセージ対応
- デバッグモードでの詳細表示
- 安全なフォールバック処理

### 依存関係
- `helper_api.py`への依存
- Streamlit環境での実行必須
- 設定ファイル（config.yaml）の存在が前提

この設計書により、Streamlit UIヘルパーライブラリの構造と使用方法が明確になり、効率的なStreamlitアプリケーション開発が可能になります。
