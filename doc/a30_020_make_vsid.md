主な改修点とクラス化
🏗️ クラス設計

VectorStoreConfig - データクラス

各データセットの設定管理（ファイル名、チャンクサイズ等）
型安全な設定管理


VectorStoreProcessor - データ処理クラス

テキストファイル読み込み
チャンク分割処理
JSONL形式変換


VectorStoreManager - API管理クラス

OpenAI Vector Store操作
作成進行状況監視
エラーハンドリング


VectorStoreUI - UI管理クラス

Streamlit画面管理
進行状況表示
結果可視化
