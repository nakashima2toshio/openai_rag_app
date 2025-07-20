# a30_011_make_rag_data_customer.py
# カスタマーサポートFAQデータのRAG前処理（シンプル版）
# streamlit run a30_011_make_rag_data_customer.py --server.port=8501

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# 共通RAGヘルパーモジュールをインポート
try:
    from helper_rag import (
        # 設定・定数
        RAGConfig,

        # 前処理関数（移行済み）
        clean_text,
        combine_columns,
        additional_preprocessing,
        validate_data,

        # ファイル・データ処理
        load_dataset,
        create_download_data,
        process_rag_data,

        # UI関連
        setup_rag_page,
        setup_sidebar_controls,
        display_statistics,
        show_usage_instructions,
    )
except ImportError as e:
    st.error(f"helper_rag.pyのインポートに失敗しました: {e}")
    st.info("helper_rag.pyが同じディレクトリにあることを確認してください。")
    st.stop()

# 既存ヘルパーモジュールもインポート
try:
    from helper_st import UIHelper, error_handler_ui
    from helper_api import logger
except ImportError as e:
    st.warning(f"既存ヘルパーモジュールの一部が利用できません: {e}")
    # 基本的なログ設定
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ==================================================
# ファイル保存関数
# ==================================================
def create_output_directory() -> Path:
    """OUTPUTディレクトリの作成"""
    try:
        import os
        current_dir = os.getcwd()

        # 相対パスでOUTPUTディレクトリを作成
        output_dir = Path("OUTPUT")
        absolute_output_dir = output_dir.resolve()

        logger.info(f"現在のディレクトリ: {current_dir}")
        logger.info(f"OUTPUT相対パス: {output_dir}")
        logger.info(f"OUTPUT絶対パス: {absolute_output_dir}")

        # ディレクトリの作成
        output_dir.mkdir(exist_ok=True)

        # ディレクトリが実際に作成されたか確認
        if not output_dir.exists():
            raise OSError(f"OUTPUTディレクトリの作成に失敗: {absolute_output_dir}")

        # 書き込み権限のテスト
        test_file = output_dir / ".test_write"
        try:
            test_file.write_text("test", encoding='utf-8')
            if test_file.exists():
                test_file.unlink()
                logger.info("書き込み権限テスト: 成功")
            else:
                raise PermissionError("テストファイルの作成に失敗")
        except Exception as e:
            raise PermissionError(f"書き込み権限テストに失敗: {e}")

        logger.info(f"OUTPUTディレクトリ準備完了: {absolute_output_dir}")
        return output_dir

    except PermissionError as e:
        logger.error(f"権限エラー: {e}")
        raise PermissionError(f"OUTPUTフォルダの作成/書き込み権限がありません: {e}")
    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        raise Exception(f"OUTPUTフォルダの作成に失敗しました: {e}")


def save_files_to_output(df_processed, dataset_type: str, csv_data: str, text_data: str = None) -> Dict[str, str]:
    """
    処理済みデータをOUTPUTフォルダに保存

    Args:
        df_processed: 処理済みDataFrame
        dataset_type: データセットタイプ
        csv_data: CSVデータ文字列
        text_data: 結合テキストデータ（オプション）

    Returns:
        Dict[str, str]: 保存されたファイルパスの辞書
    """
    try:
        # デバッグ: 現在の作業ディレクトリを確認
        import os
        current_dir = os.getcwd()
        logger.info(f"現在の作業ディレクトリ: {current_dir}")

        # OUTPUTディレクトリの作成
        output_dir = create_output_directory()
        logger.info(f"OUTPUTディレクトリパス: {output_dir.absolute()}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # データの検証
        if not csv_data or len(csv_data.strip()) == 0:
            raise ValueError("CSVデータが空です")

        logger.info(f"保存開始: {dataset_type}, {len(df_processed)}行, タイムスタンプ: {timestamp}")

        # CSVファイルの保存
        csv_filename = f"preprocessed_{dataset_type}_{len(df_processed)}rows_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        logger.info(f"CSVファイル保存開始: {csv_path}")

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_data)

        # ファイルが実際に作成されたか確認
        if csv_path.exists():
            file_size = csv_path.stat().st_size
            saved_files['csv'] = str(csv_path)
            logger.info(f"CSVファイル保存完了: {csv_path} ({file_size} bytes)")
        else:
            raise IOError(f"CSVファイルの作成に失敗: {csv_path}")

        # 結合テキストファイルの保存
        if text_data and len(text_data.strip()) > 0:
            # txt_filename = f"combined_{dataset_type}_{len(df_processed)}rows_{timestamp}.txt"
            txt_filename = f"{dataset_type}.txt"  # 固定ファイル名に変更
            txt_path = output_dir / txt_filename
            logger.info(f"テキストファイル保存開始: {txt_path}")

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_data)

            if txt_path.exists():
                file_size = txt_path.stat().st_size
                saved_files['txt'] = str(txt_path)
                logger.info(f"テキストファイル保存完了: {txt_path} ({file_size} bytes)")
            else:
                logger.warning(f"テキストファイルの作成に失敗: {txt_path}")
        else:
            logger.info("テキストデータが空のため、テキストファイルをスキップ")

        # メタデータファイルの保存
        metadata = {
            "dataset_type"        : dataset_type,
            "original_rows"       : st.session_state.get('original_rows', 0),
            "processed_rows"      : len(df_processed),
            "processing_timestamp": timestamp,
            "created_at"          : datetime.now().isoformat(),
            "working_directory"   : current_dir,
            "output_directory"    : str(output_dir.absolute()),
            "files_created"       : list(saved_files.keys())
        }

        metadata_filename = f"metadata_{dataset_type}_{timestamp}.json"
        metadata_path = output_dir / metadata_filename
        logger.info(f"メタデータファイル保存開始: {metadata_path}")

        # helper_api.pyのsafe_json_dumpsを使用
        try:
            from helper_api import safe_json_dumps
            metadata_json = safe_json_dumps(metadata)
        except ImportError:
            import json
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata_json)

        if metadata_path.exists():
            file_size = metadata_path.stat().st_size
            saved_files['metadata'] = str(metadata_path)
            logger.info(f"メタデータファイル保存完了: {metadata_path} ({file_size} bytes)")
        else:
            logger.warning(f"メタデータファイルの作成に失敗: {metadata_path}")

        logger.info(f"全ファイル保存完了: {len(saved_files)}個のファイルを保存")

        # 最終確認: すべてのファイルが存在するかチェック
        for file_type, file_path in saved_files.items():
            if not Path(file_path).exists():
                logger.error(f"ファイルが見つかりません: {file_type} - {file_path}")

        return saved_files

    except Exception as e:
        logger.error(f"ファイル保存エラー: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"エラー詳細: {traceback.format_exc()}")
        raise


# ==================================================
# カスタマーサポートFAQ特有の処理関数
# ==================================================
def validate_customer_support_data_specific(df) -> List[str]:
    """
    カスタマーサポートFAQデータ特有の検証

    Args:
        df: 検証対象のDataFrame

    Returns:
        List[str]: カスタマーサポートデータ特有の検証結果
    """
    support_issues = []

    # カスタマーサポート関連用語の存在確認
    support_keywords = [
        '問題', '解決', 'トラブル', 'エラー', 'サポート', 'ヘルプ', '対応',
        'problem', 'issue', 'error', 'help', 'support', 'solution', 'troubleshoot'
    ]

    if 'question' in df.columns:
        questions_with_support_terms = 0
        for _, row in df.iterrows():
            question_text = str(row.get('question', '')).lower()
            if any(keyword in question_text for keyword in support_keywords):
                questions_with_support_terms += 1

        support_ratio = (questions_with_support_terms / len(df)) * 100
        support_issues.append(f"サポート関連用語を含む質問: {questions_with_support_terms}件 ({support_ratio:.1f}%)")

    # 回答の長さ分析（サポート回答は実用的な長さ）
    if 'answer' in df.columns:
        answer_lengths = df['answer'].astype(str).str.len()
        avg_answer_length = answer_lengths.mean()
        if avg_answer_length < 50:
            support_issues.append(f"⚠️ 平均回答長が短い可能性: {avg_answer_length:.0f}文字")
        else:
            support_issues.append(f"✅ 適切な回答長: 平均{avg_answer_length:.0f}文字")

    # 質問の種類分析（簡易版）
    if 'question' in df.columns:
        question_starters = ['どうすれば', 'なぜ', 'いつ', 'どこで', 'どのように', 'what', 'how', 'why', 'when',
                             'where']
        question_type_count = 0
        for _, row in df.iterrows():
            question_text = str(row.get('question', '')).lower()
            if any(starter in question_text for starter in question_starters):
                question_type_count += 1

        question_type_ratio = (question_type_count / len(df)) * 100
        support_issues.append(f"疑問形質問: {question_type_count}件 ({question_type_ratio:.1f}%)")

    return support_issues


# ==================================================
# メイン処理関数
# ==================================================
@error_handler_ui
def main():
    """メイン処理関数"""

    # データセットタイプの設定
    DATASET_TYPE = "customer_support_faq"

    # ページ設定（独立実行）
    try:
        st.set_page_config(
            page_title="カスタマーサポートFAQデータ前処理",
            page_icon="💬",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        # 既に設定済みの場合は無視
        pass

    st.title("💬 カスタマーサポートFAQデータ前処理アプリ")
    st.markdown("---")

    # サイドバー設定（独立実行）
    st.sidebar.header("設定")
    combine_columns_option = st.sidebar.checkbox(
        "複数列を結合する（Vector Store用）",
        value=True,
        help="複数列を結合してRAG用テキストを作成"
    )
    show_validation = st.sidebar.checkbox(
        "データ検証を表示",
        value=True,
        help="データの品質検証結果を表示"
    )

    # カスタマーサポートデータ特有の設定
    with st.sidebar.expander("💬 サポートデータ設定", expanded=False):
        preserve_formatting = st.checkbox(
            "書式を保護",
            value=True,
            help="回答内の重要な書式（番号付きリストなど）を保護"
        )
        normalize_questions = st.checkbox(
            "質問を正規化",
            value=True,
            help="質問文の表記ゆれを統一"
        )

    # ファイルアップロード
    st.subheader("📁 データファイルのアップロード")
    uploaded_file = st.file_uploader(
        "カスタマーサポートFAQデータのCSVファイルをアップロードしてください",
        type=['csv'],
        help="question, answer の2列を含むCSVファイル"
    )

    if uploaded_file is not None:
        try:
            # ファイル情報の確認
            st.info(f"📁 ファイル: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            # セッション状態でファイル処理状況を管理
            file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"

            # ファイルが変更された場合は再読み込み
            if st.session_state.get('current_file_key') != file_key:
                # DataFrameの読み込み（load_dataset関数を使用）
                with st.spinner("ファイルを読み込み中..."):
                    df, validation_results = load_dataset(uploaded_file, DATASET_TYPE)

                # セッション状態に保存
                st.session_state['current_file_key'] = file_key
                st.session_state['original_df'] = df
                st.session_state['validation_results'] = validation_results
                st.session_state['original_rows'] = len(df)
                st.session_state['file_processed'] = False

                logger.info(f"新しいファイルを読み込み: {len(df)}行")
            else:
                # セッション状態から取得
                df = st.session_state['original_df']
                validation_results = st.session_state['validation_results']
                logger.info(f"セッション状態からファイルを取得: {len(df)}行")

            st.success(f"ファイルが正常に読み込まれました。行数: {len(df)}")

            # 元データの表示
            st.subheader("📋 元データプレビュー")
            st.dataframe(df.head(10))

            # データ検証結果の表示
            if show_validation:
                st.subheader("🔍 データ検証")

                # 基本検証結果
                for issue in validation_results:
                    st.info(issue)

                # カスタマーサポートデータ特有の検証
                support_issues = validate_customer_support_data_specific(df)
                if support_issues:
                    st.write("**カスタマーサポートデータ特有の分析:**")
                    for issue in support_issues:
                        st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            if st.button("前処理を実行", type="primary", key="process_button"):
                try:
                    with st.spinner("前処理中..."):
                        # RAGデータの前処理
                        df_processed = process_rag_data(
                            df.copy(),  # コピーを作成して元データを保護
                            DATASET_TYPE,
                            combine_columns_option
                        )

                    st.success("前処理が完了しました！")

                    # セッション状態に処理済みデータを保存
                    st.session_state['processed_df'] = df_processed
                    st.session_state['file_processed'] = True

                    # 前処理後のデータ表示
                    st.subheader("✅ 前処理後のデータプレビュー")
                    st.dataframe(df_processed.head(10))

                    # 統計情報の表示
                    display_statistics(df, df_processed, DATASET_TYPE)

                    # カスタマーサポートデータ特有の後処理分析
                    if 'Combined_Text' in df_processed.columns:
                        st.subheader("💬 カスタマーサポートデータ特有の分析")

                        # 結合テキストのサポート用語分析
                        combined_texts = df_processed['Combined_Text']
                        support_keywords = ['問題', 'エラー', 'トラブル', 'サポート', 'ヘルプ']

                        keyword_counts = {}
                        for keyword in support_keywords:
                            count = combined_texts.str.contains(keyword, case=False).sum()
                            keyword_counts[keyword] = count

                        if keyword_counts:
                            st.write("**サポート関連用語の出現頻度:**")
                            for keyword, count in keyword_counts.items():
                                percentage = (count / len(df_processed)) * 100
                                st.write(f"- {keyword}: {count}件 ({percentage:.1f}%)")

                        # 質問の長さ分布
                        if 'question' in df_processed.columns:
                            question_lengths = df_processed['question'].str.len()
                            st.write("**質問の長さ統計:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("平均質問長", f"{question_lengths.mean():.0f}文字")
                            with col2:
                                st.metric("最長質問", f"{question_lengths.max()}文字")
                            with col3:
                                st.metric("最短質問", f"{question_lengths.min()}文字")

                    logger.info(f"カスタマーサポートFAQデータ処理完了: {len(df)} → {len(df_processed)}行")

                except Exception as process_error:
                    st.error(f"前処理エラー: {str(process_error)}")
                    logger.error(f"前処理エラー: {process_error}")

                    with st.expander("🔧 前処理エラー詳細", expanded=False):
                        import traceback
                        st.code(traceback.format_exc())

            # 処理済みデータがある場合のみダウンロード・保存セクションを表示
            if st.session_state.get('file_processed', False) and 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']

                # ダウンロード・保存セクション
                st.subheader("💾 ダウンロード・保存")

                # ダウンロード用データの作成（キャッシュ）
                if 'download_data' not in st.session_state or st.session_state.get('download_data_key') != file_key:
                    with st.spinner("ダウンロード用データを準備中..."):
                        csv_data, text_data = create_download_data(
                            df_processed,
                            combine_columns_option,
                            DATASET_TYPE
                        )
                        st.session_state['download_data'] = (csv_data, text_data)
                        st.session_state['download_data_key'] = file_key
                else:
                    csv_data, text_data = st.session_state['download_data']

                # ブラウザダウンロード
                st.write("**📥 ブラウザダウンロード**")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📊 CSV形式でダウンロード",
                        data=csv_data,
                        file_name=f"preprocessed_{DATASET_TYPE}_{len(df_processed)}rows.csv",
                        mime="text/csv",
                        help="前処理済みのカスタマーサポートFAQデータをCSV形式でダウンロード"
                    )

                with col2:
                    if text_data:
                        st.download_button(
                            label="📝 テキスト形式でダウンロード",
                            data=text_data,
                            # file_name=f"combined_{DATASET_TYPE}_{len(df_processed)}rows.txt",
                            file_name=f"customer_support_faq.txt",
                            mime="text/plain",
                            help="Vector Store/RAG用に最適化された結合テキスト"
                        )

                # ローカル保存
                st.write("**💾 ローカルファイル保存（OUTPUTフォルダ）**")

                if st.button("🔄 OUTPUTフォルダに保存", type="secondary", key="save_button"):
                    try:
                        with st.spinner("ファイル保存中..."):
                            logger.info("=== ファイル保存処理開始 ===")
                            saved_files = save_files_to_output(
                                df_processed,
                                DATASET_TYPE,
                                csv_data,
                                text_data
                            )
                            logger.info(f"=== ファイル保存処理完了: {saved_files} ===")

                        if saved_files:
                            st.success("✅ ファイル保存完了！")

                            # 保存されたファイル一覧を表示
                            with st.expander("📂 保存されたファイル一覧", expanded=True):
                                for file_type, file_path in saved_files.items():
                                    if Path(file_path).exists():
                                        file_size = Path(file_path).stat().st_size
                                        st.write(f"**{file_type.upper()}**: `{file_path}` ({file_size:,} bytes) ✅")
                                    else:
                                        st.write(
                                            f"**{file_type.upper()}**: `{file_path}` ❌ ファイルが見つかりません")

                                # OUTPUTフォルダの場所を表示
                                output_path = Path("OUTPUT").resolve()
                                st.write(f"**保存場所**: `{output_path}`")
                                file_count = len(list(output_path.glob("*")))
                                st.write(f"**フォルダ内ファイル数**: {file_count}個")
                        else:
                            st.error("❌ ファイル保存に失敗しました")

                    except Exception as save_error:
                        st.error(f"❌ ファイル保存エラー: {str(save_error)}")
                        logger.error(f"保存エラー: {save_error}")

                        with st.expander("🔧 保存エラー詳細", expanded=False):
                            import traceback
                            st.code(traceback.format_exc())

                # 既存の保存済みファイル一覧
                try:
                    output_dir = Path("OUTPUT")
                    if output_dir.exists():
                        saved_files_list = list(output_dir.glob(f"*{DATASET_TYPE}*"))
                        if saved_files_list:
                            st.write("**📁 既存の保存済みファイル**")
                            with st.expander("保存済みファイル一覧", expanded=False):
                                for file_path in sorted(saved_files_list, reverse=True)[:5]:  # 最新5件
                                    file_stats = file_path.stat()
                                    file_time = datetime.fromtimestamp(file_stats.st_mtime).strftime(
                                        "%Y-%m-%d %H:%M:%S")
                                    st.write(f"- `{file_path.name}` ({file_stats.st_size:,} bytes, {file_time})")
                except Exception as list_error:
                    logger.warning(f"ファイル一覧取得エラー: {list_error}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"ファイル読み込みエラー: {e}")

            # 詳細なエラー診断
            with st.expander("🔧 詳細エラー情報", expanded=True):
                import traceback
                st.code(traceback.format_exc())

                # ファイル情報の詳細確認
                st.write("**ファイル診断:**")
                if uploaded_file is not None:
                    st.write(f"- ファイル名: {uploaded_file.name}")
                    st.write(f"- ファイルサイズ: {uploaded_file.size:,} bytes")
                    st.write(f"- ファイルタイプ: {uploaded_file.type}")
                    st.write(f"- ファイルオブジェクト型: {type(uploaded_file)}")

                    # ファイル内容のテスト読み込み
                    try:
                        uploaded_file.seek(0)  # ファイルポインタをリセット
                        sample_content = uploaded_file.read(200).decode('utf-8', errors='ignore')
                        st.write("**ファイル先頭（200文字）:**")
                        st.code(sample_content)
                        uploaded_file.seek(0)  # リセット
                    except Exception as read_error:
                        st.write(f"ファイル読み込みテストエラー: {read_error}")
                else:
                    st.write("- ファイル: None")

                # 環境情報
                st.write("**環境情報:**")
                st.write(f"- pandas version: {pd.__version__}")
                st.write(f"- streamlit version: {st.__version__}")

    else:
        st.info("👆 CSVファイルをアップロードしてください")

    # 使用方法の説明
    show_usage_instructions(DATASET_TYPE)

    # セッション状態の表示（デバッグ用）
    if st.sidebar.checkbox("🔧 セッション状態を表示", value=False):
        with st.sidebar.expander("セッション状態", expanded=False):
            # 基本情報
            st.write(f"**現在のファイルキー**: {st.session_state.get('current_file_key', 'None')}")
            st.write(f"**ファイル処理済み**: {st.session_state.get('file_processed', False)}")

            if 'original_df' in st.session_state:
                df = st.session_state['original_df']
                st.write(f"**元データ**: {len(df)}行")

            if 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']
                st.write(f"**処理済みデータ**: {len(df_processed)}行")

            # OUTPUTフォルダの状態
            try:
                output_dir = Path("OUTPUT")
                if output_dir.exists():
                    file_count = len(list(output_dir.glob(f"*{DATASET_TYPE}*")))
                    st.write(f"**保存済みファイル**: {file_count}個")
                    st.write(f"**フォルダパス**: `{output_dir.resolve()}`")
                else:
                    st.write("**OUTPUTフォルダ**: 未作成")
            except Exception as e:
                st.write(f"**フォルダ状態**: エラー ({e})")


# ==================================================
# アプリケーション実行
# ==================================================
if __name__ == "__main__":
    main()

# 実行コマンド:
# streamlit run a30_011_make_rag_data_customer.py --server.port=8501
