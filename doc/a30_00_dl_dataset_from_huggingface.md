# RAGデータセット処理プログラム詳細設計書

## 概要書

### 処理の概要
本プログラムは、OpenAI APIを使用したRAG（Retrieval-Augmented Generation）システムの学習用データセットを、Hugging Faceから自動的にダウンロード・管理するプログラムです。

5つの異なるドメインのQAデータセット（カスタマーサポート、一般知識、医療、科学、法律）を対象とし、これらのデータセットをCSV形式でローカルに保存・管理します。

**対象データセット：**
1. カスタマーサポート・FAQデータセット（MakTek/Customer_support_faqs_dataset）
2. 一般知識・トリビアQAデータセット（trivia_qa）
3. 医療質問回答データセット（FreedomIntelligence/medical-o1-reasoning-SFT）
4. 科学・技術QAデータセット（sciq）
5. 法律・判例QAデータセット（nguha/legalbench）

### mainの処理の流れ
1. **データセット存在チェック**
   - `datasets`ディレクトリの存在確認
   - CSVファイルの存在確認

2. **条件分岐処理**
   - データセットが存在しない場合：ダウンロード処理を実行
   - データセットが存在する場合：既存データセット使用メッセージを表示

3. **データセット表示処理**
   - ダウンロード済みデータセットの内容を表示

---

## 関数一覧

| 関数名 | 処理概要 |
|--------|----------|
| `download_dataset()` | Hugging Faceから指定された5つのデータセットをダウンロードし、CSV形式で保存する |
| `show_dataset()` | カスタマーサポートFAQデータセットの詳細情報（データ数、カラム、先頭10行）を表示する |
| `main()` | プログラムのメイン処理。データセット存在チェック、ダウンロード、表示を統合管理する |

---

## 関数の詳細設計書

### 1. download_dataset()

#### 処理概要
グローバル変数`datasets_to_download`で定義された5つのデータセット設定情報を基に、Hugging Faceからデータセットをダウンロードし、pandas DataFrameを経由してCSV形式で保存する。

#### 処理の流れ
1. `datasets`ディレクトリを作成（存在しない場合）
2. `datasets_to_download`リストをループ処理
3. 各データセットに対して：
   - `load_dataset()`でHugging Faceからダウンロード
   - pandas DataFrameに変換
   - CSV形式で`datasets/<name>.csv`に保存
   - 成功・失敗メッセージを表示
4. 全体の完了メッセージを表示

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - グローバル変数`datasets_to_download`（データセット設定情報のリスト）
  - インターネット接続（Hugging Faceからのダウンロード用）

- **PROCESS：**
  - ディレクトリ作成処理
  - Hugging Face Datasetsライブラリを使用したダウンロード処理
  - pandas DataFrameへの変換処理
  - CSV形式での保存処理
  - 例外処理（エラーハンドリング）

- **OUTPUT：**
  - `datasets/`ディレクトリ内のCSVファイル群
    - `customer_support_faq.csv`
    - `trivia_qa.csv`
    - `medical_qa.csv`
    - `sciq_qa.csv`
    - `legal_qa.csv`
  - コンソール出力（進捗状況、エラーメッセージ）

---

### 2. show_dataset()

#### 処理概要
カスタマーサポートFAQデータセット（`customer_support_faq.csv`）の基本情報を読み込み、データの概要と先頭10行を表示する。

#### 処理の流れ
1. `datasets/customer_support_faq.csv`のパス設定
2. ファイル存在確認
3. ファイルが存在しない場合：エラーメッセージと解決策を表示
4. ファイルが存在する場合：
   - pandas.read_csv()でCSVファイルを読み込み
   - データセット名、データ数、カラム一覧を表示
   - 先頭10行のデータを表示
5. 例外処理（読み込みエラーの場合）

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - `datasets/customer_support_faq.csv`ファイル

- **PROCESS：**
  - ファイル存在確認処理
  - pandas CSV読み込み処理
  - データフレーム情報抽出処理
  - 例外処理

- **OUTPUT：**
  - コンソール出力
    - データセット名
    - データ数（行数）
    - カラム一覧
    - 先頭10行のデータ表示
    - エラーメッセージ（該当する場合）

---

### 3. main()

#### 処理概要
プログラム全体の制御を行うメイン関数。データセットの存在チェックを行い、必要に応じてダウンロード処理を実行し、最終的にデータセットの表示を行う。

#### 処理の流れ
1. `datasets`ディレクトリのパス設定
2. データセット存在チェック
   - ディレクトリの存在確認
   - CSVファイルの存在確認（`*.csv`パターン）
3. 条件分岐処理
   - データセットが存在しない場合：
     - ダウンロード開始メッセージを表示
     - `download_dataset()`を呼び出し
   - データセットが存在する場合：
     - 既存データセット使用メッセージを表示
4. `show_dataset()`を呼び出してデータセット表示

#### IPO（INPUT、PROCESS、OUTPUT）
- **INPUT：**
  - ファイルシステム（`datasets`ディレクトリとCSVファイル群）

- **PROCESS：**
  - ディレクトリ・ファイル存在確認処理
  - 条件分岐処理
  - 他関数の呼び出し制御処理

- **OUTPUT：**
  - コンソール出力（状況メッセージ）
  - `download_dataset()`および`show_dataset()`の実行結果

---

## 補足情報

### 使用ライブラリ
- `datasets`：Hugging Face Datasetsライブラリ
- `pandas`：データフレーム操作・CSV処理
- `pathlib`：パス操作
- `typing`：型ヒント

### 設定データ構造
`datasets_to_download`は辞書のリストで、各データセットの設定情報を格納：
- `name`：ローカル保存用の名前
- `hfpath`：Hugging Faceでのデータセットパス
- `config`：データセット設定（None可）
- `split`：データ分割指定（train, test等）

### 注意事項
- 現在`show_dataset()`はカスタマーサポートFAQデータセットのみに対応
- 他の4つのデータセット表示機能は未実装
- インターネット接続が必要（Hugging Faceからのダウンロード時）
