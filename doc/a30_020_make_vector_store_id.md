# a30_020_make_vector_store_id.py 設計書

## 1. 概要書

### 1.1 処理の概要

4つの異なるデータセット（カスタマーサポートFAQ、医療QA、科学・技術QA、法律・判例QA）のテキストファイルを読み込み、OpenAI Vector Storeに登録するバッチ処理プログラム。各データセットを適切なチャンクサイズで分割し、JSONL形式に変換してOpenAI APIを通じてVector Storeを作成する。

**主要機能:**

- 複数データセットのバッチ処理
- テキストの自動チャンク分割
- OpenAI Vector Store自動作成
- 処理結果の保存・管理
- エラーハンドリングと進捗表示

### 1.2 mainの処理の流れ

環境確認
├─ OPENAI_API_KEY環境変数確認
└─ 出力ディレクトリ存在確認

必要ファイル確認
├─ 4つのデータセットファイル存在確認
├─ customer_support_faq.txt
├─ medical_qa.txt
├─ sciq_qa.txt
└─ legal_qa.txt

Vector Store Manager初期化
└─ OpenAIクライアント準備

全データセット処理
├─ 各データセットを順次処理
├─ ファイル読み込み
├─ JSONL形式変換
├─ Vector Store作成
└─ 結果収集

結果保存・表示
├─ vector_store_ids.json保存
├─ 処理結果サマリー表示
└─ 既存Vector Store一覧表示


エラーハンドリング
├─ 各段階での例外処理
└─ ユーザー中断対応

## 2. 関数一覧

### 2.1 テキスト処理関数


| 関数名                 | 処理概要                         |
| ---------------------- | -------------------------------- |
| `load_text_file()`     | テキストファイル読み込み・前処理 |
| `chunk_text()`         | 長文テキストのチャンク分割       |
| `text_to_jsonl_data()` | テキスト行のJSONL形式変換        |

### 2.2 OpenAI API関数


| 関数名                                  | 処理概要                        |
| --------------------------------------- | ------------------------------- |
| `get_embeddings_batch()`                | バッチでのEmbedding生成         |
| `create_vector_store_from_jsonl_data()` | JSONLデータからVector Store作成 |

### 2.3 Vector Store管理クラス


| メソッド名                                  | 処理概要                 |
| ------------------------------------------- | ------------------------ |
| `VectorStoreManager.__init__()`             | マネージャー初期化       |
| `VectorStoreManager.process_all_datasets()` | 全データセット一括処理   |
| `VectorStoreManager._print_summary()`       | 処理結果サマリー表示     |
| `VectorStoreManager.save_results()`         | 結果のJSON保存           |
| `VectorStoreManager.list_vector_stores()`   | 既存Vector Store一覧取得 |

### 2.4 メイン・ユーティリティ関数


| 関数名                         | 処理概要             |
| ------------------------------ | -------------------- |
| `main()`                       | メイン実行制御       |
| `create_single_vector_store()` | 単一データセット処理 |

## 3. 関数の詳細設計書

### 3.1 load_text_file()

**処理概要:**
指定されたテキストファイルを読み込み、空行や短すぎる行を除去して有効なテキスト行のリストを返す。

**処理の流れ:**

ファイル存在確認
UTF-8エンコーディングでファイル読み込み

各行の前処理
├─ 先頭・末尾空白除去
├─ 空行除去
└─ 10文字未満の短い行除去


クリーンな行リスト作成
読み込み結果ログ出力

**IPO:**

- **INPUT:** `filepath` (Path) - 読み込み対象ファイルパス
- **PROCESS:** ファイル読み込み、行フィルタリング、クリーニング
- **OUTPUT:** `List[str]` - 有効なテキスト行のリスト

### 3.2 chunk_text()

**処理概要:**
長いテキストを指定されたサイズのチャンクに分割し、文の境界を考慮した自然な分割を行う。

**処理の流れ:**

テキスト長チェック
└─ chunk_size以下の場合はそのまま返却

チャンク分割ループ
├─ 開始位置から終了位置計算
├─ 文境界調整
│   ├─ 句読点位置検索
│   └─ 自然な区切り位置決定
├─ チャンク抽出・クリーニング
└─ 次の開始位置計算（オーバーラップ考慮）


無限ループ防止チェック
分割チャンクリスト返却

**IPO:**

- **INPUT:**
  - `text` (str) - 分割対象テキスト
  - `chunk_size` (int) - チャンクサイズ（デフォルト1000）
  - `overlap` (int) - オーバーラップサイズ（デフォルト100）
- **PROCESS:** テキスト分割、文境界調整、オーバーラップ処理
- **OUTPUT:** `List[str]` - 分割されたテキストチャンクリスト

### 3.3 text_to_jsonl_data()

**処理概要:**
テキスト行をJSONL形式のデータ構造に変換し、メタデータとIDを付与してVector Store用に準備する。

**処理の流れ:**

データセット設定取得
├─ chunk_size取得
└─ overlap取得

各テキスト行の処理
├─ clean_text()によるクリーニング
├─ 空文字チェック
└─ chunk_text()による分割

チャンクごとのJSONLエントリ作成
├─ 一意ID生成（dataset_type_行番号_チャンク番号）
├─ テキスト内容設定
└─ メタデータ設定


変換統計ログ出力
JSONLデータリスト返却

**IPO:**

- **INPUT:**
  - `lines` (List[str]) - テキスト行リスト
  - `dataset_type` (str) - データセットタイプ
- **PROCESS:** テキストクリーニング、チャンク分割、JSONL構造化
- **OUTPUT:** `List[Dict[str, str]]` - JSONL用データ構造リスト

### 3.4 create_vector_store_from_jsonl_data()

**処理概要:**
JSONL形式のデータからOpenAI Vector Storeを作成し、ファイルアップロード、Vector Store作成、ファイル関連付けを行う。

**処理の流れ:**

一時ファイル作成
├─ tempfile.NamedTemporaryFile使用
├─ JSONLエントリの書き込み
└─ UTF-8エンコーディング

OpenAIファイルアップロード
├─ files.create API呼び出し
├─ purpose="assistants"設定
└─ アップロードファイルID取得

Vector Store作成
├─ vector_stores.create API呼び出し
├─ メタデータ設定
└─ Vector Store ID取得

ファイル関連付け
├─ vector_stores.files.create呼び出し
└─ Vector Storeとファイルのリンク

処理完了待機
├─ 最大10分間の待機
├─ 5秒間隔のステータスチェック
├─ completed/failed/in_progress判定
└─ タイムアウト処理


結果確認・ログ出力
一時ファイル削除

**IPO:**

- **INPUT:**
  - `jsonl_data` (List[Dict]) - JSONL形式データリスト
  - `store_name` (str) - Vector Store名
- **PROCESS:** ファイル作成、OpenAI API呼び出し、状態監視
- **OUTPUT:** `Optional[str]` - Vector Store ID（成功時）またはNone（失敗時）

### 3.5 VectorStoreManager.process_all_datasets()

**処理概要:**
全データセットを順次処理してVector Storeを作成し、結果を管理する統括メソッド。

**処理の流れ:**

処理開始ログ出力
各データセットのループ処理
├─ ファイルパス構築
├─ ファイル存在確認
├─ load_text_file()呼び出し
├─ text_to_jsonl_data()呼び出し
├─ create_vector_store_from_jsonl_data()呼び出し
├─ 結果記録（成功/失敗）
└─ エラーハンドリング


結果辞書構築
_print_summary()による結果表示
結果辞書返却

**IPO:**

- **INPUT:** `output_dir` (Path) - 出力ディレクトリ（デフォルト: OUTPUT_DIR）
- **PROCESS:** 全データセット順次処理、エラーハンドリング、結果集約
- **OUTPUT:** `Dict[str, Optional[str]]` - データセット名→Vector Store IDマッピング

### 3.6 main()

**処理概要:**
プログラム全体の実行制御を行い、環境確認からVector Store作成、結果保存まで一連の処理を統括する。

**処理の流れ:**

プログラム開始ログ

- 環境確認
  ├─ OPENAI_API_KEY環境変数確認
  └─ 未設定時はエラー終了
- 出力ディレクトリ確認
  ├─ OUTPUT_DIR存在確認
  └─ 不存在時は必要ファイル一覧表示して終了
- 必要ファイル存在確認
  ├─ 4つのデータセットファイルチェック
  └─ 不足ファイルがある場合はエラー終了
- Vector Store Manager初期化
- メイン処理実行
  ├─ process_all_datasets()呼び出し
  ├─ save_results()による結果保存
  └─ list_vector_stores()による一覧表示
- 　
  例外処理
  ├─ KeyboardInterrupt対応
  └─ 予期しないエラー処理
- 　
  プログラム終了ログ

**IPO:**

- **INPUT:** なし（環境変数、ファイルシステムから取得）
- **PROCESS:** 環境確認、全データセット処理、結果保存、エラーハンドリング
- **OUTPUT:** なし（副作用: Vector Store作成、ファイル保存、ログ出力）

## 4. 不足情報の指摘

### 4.1 設定・仕様の不足

1. **API制限・制約**

   - OpenAI APIのレート制限対応詳細
   - 最大ファイルサイズ制限
   - Vector Store数の上限
   - 同時処理可能数の制限
2. **チャンク分割仕様**

   - 各データセットの最適チャンクサイズ根拠
   - オーバーラップサイズの決定基準
   - 文境界検出のルール詳細
   - 多言語対応の有無
3. **メタデータ仕様**

   - Vector Storeメタデータの項目定義
   - 検索時に利用するメタデータ
   - メタデータの文字数制限

### 4.2 エラーハンドリング仕様の不足

1. **API エラー対応**

   - 各種APIエラーコードと対応策
   - リトライ回数・間隔の設定
   - 部分失敗時の継続/中断判定
   - クオータ超過時の対応手順
2. **ファイル処理エラー**

   - 大容量ファイル処理時のメモリ管理
   - 文字エンコーディングエラー対応
   - 一時ファイル作成失敗時の処理
   - ディスク容量不足時の対応

### 4.3 パフォーマンス仕様の不足

1. **処理性能要件**

   - 各データセットの処理時間目安
   - メモリ使用量の上限
   - 並行処理の可否
   - 大容量データ対応方法
2. **監視・ログ仕様**

   - 進捗監視の詳細度
   - ログレベルの設定方法
   - エラーログの出力先
   - 処理統計の収集項目

### 4.4 運用・保守仕様の不足

1. **設定管理**

   - 外部設定ファイルの利用可否
   - 環境別設定の切り替え方法
   - デフォルト値の変更手順
   - 設定値の検証方法
2. **データ管理**

   - Vector Store の更新・削除手順
   - バックアップ・リストア方法
   - データ整合性チェック
   - 古いVector Storeの削除ポリシー

### 4.5 セキュリティ仕様の不足

1. **APIキー管理**

   - APIキーの安全な保管方法
   - キーローテーション手順
   - アクセス権限の制御
   - 監査ログの要件
2. **データ保護**

   - 一時ファイルの暗号化要否
   - 個人情報の検出・保護
   - データ削除の確実性
   - 通信の暗号化要件
