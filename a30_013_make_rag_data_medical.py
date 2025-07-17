# a30_013_make_rag_data_medical.py
# 医療QAデータのRAG前処理（シンプル版）
# streamlit run a30_013_make_rag_data_medical.py --server.port=8503

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
            txt_filename = f"combined_{dataset_type}_{len(df_processed)}rows_{timestamp}.txt"
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
# 医療QA特有の処理関数
# ==================================================
def validate_medical_data_specific(df) -> List[str]:
    """
    医療QAデータ特有の検証

    Args:
        df: 検証対象のDataFrame

    Returns:
        List[str]: 医療データ特有の検証結果
    """
    medical_issues = []

    # 医療用語の存在確認（簡易版）
    medical_keywords = [
        '症状', '診断', '治療', '薬', '病気', '疾患', '患者',
        'symptom', 'diagnosis', 'treatment', 'medicine', 'disease', 'patient'
    ]

    if 'Question' in df.columns:
        questions_with_medical_terms = 0
        for _, row in df.iterrows():
            question_text = str(row.get('Question', '')).lower()
            if any(keyword in question_text for keyword in medical_keywords):
                questions_with_medical_terms += 1

        medical_ratio = (questions_with_medical_terms / len(df)) * 100
        medical_issues.append(f"医療関連用語を含む質問: {questions_with_medical_terms}件 ({medical_ratio:.1f}%)")

    # 回答の長さ分析（医療回答は通常詳細）
    if 'Response' in df.columns:
        response_lengths = df['Response'].astype(str).str.len()
        avg_response_length = response_lengths.mean()
        if avg_response_length < 100:
            medical_issues.append(f"⚠️ 平均回答長が短い可能性: {avg_response_length:.0f}文字")
        else:
            medical_issues.append(f"✅ 適切な回答長: 平均{avg_response_length:.0f}文字")

    return medical_issues


def display_medical_specific_info():
    """医療データ特有の情報を表示"""
    with st.sidebar.expander("🏥 医療データ特有設定", expanded=False):
        st.write("**医療データの特徴:**")
        st.write("- 専門用語の多用")
        st.write("- 詳細な説明文")
        st.write("- 正確性が重要")

        st.write("**前処理の注意点:**")
        st.write("- 医療用語の保持")
        st.write("- 文脈の保持")
        st.write("- 略語の展開")

        # 医療データ処理のオプション
        preserve_medical_terms = st.checkbox(
            "医療用語を保護",
            value=True,
            help="医療専門用語の過度な正規化を防ぐ"
        )

        expand_abbreviations = st.checkbox(
            "略語を展開",
            value=False,
            help="一般的な医療略語を展開形に変換"
        )

        return preserve_medical_terms, expand_abbreviations


# ==================================================
# メイン処理関数
# ==================================================
@error_handler_ui
def main():
    """メイン処理関数"""

    # データセットタイプの設定
    DATASET_TYPE = "medical_qa"

    # ページ設定（独立実行）
    try:
        st.set_page_config(
            page_title="医療QAデータ前処理",
            page_icon="🏥",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        # 既に設定済みの場合は無視
        pass

    st.title("🏥 医療QAデータ前処理アプリ")
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

    # 医療データ特有の設定（簡略化）
    with st.sidebar.expander("🏥 医療データ設定", expanded=False):
        preserve_medical_terms = st.checkbox(
            "医療用語を保護",
            value=True,
            help="医療専門用語の過度な正規化を防ぐ"
        )
        expand_abbreviations = st.checkbox(
            "略語を展開",
            value=False,
            help="一般的な医療略語を展開形に変換"
        )

    # ファイルアップロード
    st.subheader("📁 データファイルのアップロード")
    uploaded_file = st.file_uploader(
        "医療QAデータのCSVファイルをアップロードしてください",
        type=['csv'],
        help="Question, Complex_CoT, Response の3列を含むCSVファイル",
        key="medical_qa_uploader"
    )
    # ファイルアップロード
    st.subheader("📁 データファイルのアップロード")
    uploaded_file = st.file_uploader(
        "医療QAデータのCSVファイルをアップロードしてください",
        type=['csv'],
        help="Question, Complex_CoT, Response の3列を含むCSVファイル"
    )

    if uploaded_file is not None:
        try:
            # ファイル情報の確認
            st.info(f"📁 ファイル: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            # DataFrameの読み込み
            df = pd.read_csv(uploaded_file)

            # 基本検証
            validation_results = validate_data(df, DATASET_TYPE)

            logger.info(f"データセット読み込み完了: {len(df)}行, {len(df.columns)}列")
            st.success(f"ファイルが正常に読み込まれました。行数: {len(df)}")

            # 元データの表示
            st.subheader("📋 元データプレビュー")
            st.dataframe(df.head(10))

            # データ検証結果の表示
            if show_validation:
                st.subheader("🔍 データ検証")
                for issue in validation_results:
                    st.info(issue)

                # 医療データ特有の検証
                medical_issues = validate_medical_data_specific(df)
                if medical_issues:
                    st.write("**医療データ特有の分析:**")
                    for issue in medical_issues:
                        st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            if st.button("前処理を実行", type="primary"):
                try:
                    with st.spinner("前処理中..."):
                        # RAGデータの前処理
                        df_processed = process_rag_data(
                            df,
                            DATASET_TYPE,
                            combine_columns_option
                        )

                    st.success("前処理が完了しました！")

                    # 前処理後のデータ表示
                    st.subheader("✅ 前処理後のデータプレビュー")
                    st.dataframe(df_processed.head(10))

                    # 統計情報の表示
                    display_statistics(df, df_processed, DATASET_TYPE)

                    # 医療データ特有の後処理分析
                    if 'Combined_Text' in df_processed.columns:
                        st.subheader("🏥 医療データ特有の分析")

                        # 結合テキストの医療用語分析
                        combined_texts = df_processed['Combined_Text']
                        medical_keywords = ['症状', '診断', '治療', '薬', '病気', '疾患']

                        keyword_counts = {}
                        for keyword in medical_keywords:
                            count = combined_texts.str.contains(keyword, case=False).sum()
                            keyword_counts[keyword] = count

                        if keyword_counts:
                            st.write("**医療用語の出現頻度:**")
                            for keyword, count in keyword_counts.items():
                                percentage = (count / len(df_processed)) * 100
                                st.write(f"- {keyword}: {count}件 ({percentage:.1f}%)")

                    # ダウンロード・保存セクション
                    st.subheader("💾 ダウンロード・保存")

                    # ダウンロード用データの作成
                    csv_data, text_data = create_download_data(
                        df_processed,
                        combine_columns_option,
                        DATASET_TYPE
                    )

                    # ブラウザダウンロード
                    st.write("**📥 ブラウザダウンロード**")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="📊 CSV形式でダウンロード",
                            data=csv_data,
                            file_name=f"preprocessed_{DATASET_TYPE}_{len(df_processed)}rows.csv",
                            mime="text/csv",
                            help="前処理済みの医療QAデータをCSV形式でダウンロード"
                        )

                    with col2:
                        if text_data:
                            st.download_button(
                                label="📝 テキスト形式でダウンロード",
                                data=text_data,
                                file_name=f"combined_{DATASET_TYPE}_{len(df_processed)}rows.txt",
                                mime="text/plain",
                                help="Vector Store/RAG用に最適化された結合テキスト"
                            )

                    # ローカル保存
                    st.write("**💾 ローカルファイル保存（OUTPUTフォルダ）**")

                    if st.button("🔄 OUTPUTフォルダに保存", type="secondary"):
                        try:
                            with st.spinner("ファイル保存中..."):
                                saved_files = save_files_to_output(
                                    df_processed,
                                    DATASET_TYPE,
                                    csv_data,
                                    text_data
                                )

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

                    logger.info(f"医療QAデータ処理完了: {len(df)} → {len(df_processed)}行")

                except Exception as process_error:
                    st.error(f"前処理エラー: {str(process_error)}")
                    logger.error(f"前処理エラー: {process_error}")

                    with st.expander("🔧 前処理エラー詳細", expanded=False):
                        import traceback
                        st.code(traceback.format_exc())

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
                import pandas as pd
                st.write(f"- pandas version: {pd.__version__}")
                st.write(f"- streamlit version: {st.__version__}")

    else:
        st.info("👆 CSVファイルをアップロードしてください")
        try:
            # データセットの読み込みと基本検証
            df, validation_results = load_dataset(uploaded_file, DATASET_TYPE)

            # ファイルアップロードの際に元の行数を保存
            st.session_state['original_rows'] = len(df)
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

                # 医療データ特有の検証
                medical_issues = validate_medical_data_specific(df)
                if medical_issues:
                    st.write("**医療データ特有の分析:**")
                    for issue in medical_issues:
                        st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            if st.button("前処理を実行", type="primary"):
                # RAGデータの前処理
                df_processed = process_rag_data(
                    df,
                    DATASET_TYPE,
                    combine_columns_option
                )

                st.success("前処理が完了しました！")

                # 前処理後のデータ表示
                st.subheader("✅ 前処理後のデータプレビュー")
                st.dataframe(df_processed.head(10))

                # 統計情報の表示
                display_statistics(df, df_processed, DATASET_TYPE)

                # 医療データ特有の後処理分析
                if 'Combined_Text' in df_processed.columns:
                    st.subheader("🏥 医療データ特有の分析")

                    # 結合テキストの医療用語分析
                    combined_texts = df_processed['Combined_Text']
                    medical_keywords = ['症状', '診断', '治療', '薬', '病気', '疾患']

                    keyword_counts = {}
                    for keyword in medical_keywords:
                        count = combined_texts.str.contains(keyword, case=False).sum()
                        keyword_counts[keyword] = count

                    if keyword_counts:
                        st.write("**医療用語の出現頻度:**")
                        for keyword, count in keyword_counts.items():
                            percentage = (count / len(df_processed)) * 100
                            st.write(f"- {keyword}: {count}件 ({percentage:.1f}%)")

                # ダウンロード機能
                st.subheader("💾 ダウンロード・保存")

                # ダウンロード用データの作成
                csv_data, text_data = create_download_data(
                    df_processed,
                    combine_columns_option,
                    DATASET_TYPE
                )

                # ダウンロード機能（ブラウザ経由）
                st.write("**📥 ブラウザダウンロード**")
                col1, col2 = st.columns(2)

                with col1:
                    # CSVダウンロード
                    st.download_button(
                        label="📊 CSV形式でダウンロード",
                        data=csv_data,
                        file_name=f"preprocessed_{DATASET_TYPE}_{str(len(df_processed))}rows.csv",
                        mime="text/csv",
                        help="前処理済みの医療QAデータをCSV形式でダウンロード"
                    )

                with col2:
                    # 結合テキストダウンロード
                    if text_data:
                        st.download_button(
                            label="📝 テキスト形式でダウンロード",
                            data=text_data,
                            file_name=f"combined_{DATASET_TYPE}_{str(len(df_processed))}rows.txt",
                            mime="text/plain",
                            help="Vector Store/RAG用に最適化された結合テキスト"
                        )

                # ローカルファイル保存機能（OUTPUTフォルダ）
                st.write("**💾 ローカルファイル保存（OUTPUTフォルダ）**")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info("💡 ファイルはプロジェクトのOUTPUTフォルダに保存されます")

                with col2:
                    if st.button("🔄 OUTPUTフォルダに保存", type="primary"):
                        # ボタンクリック確認
                        st.write("🔄 保存処理を開始します...")
                        logger.info("=== 保存ボタンがクリックされました ===")

                        try:
                            # デバッグ: データの存在確認
                            logger.info(f"csv_data存在確認: {csv_data is not None}")
                            logger.info(f"csv_data長さ: {len(csv_data) if csv_data else 0}")
                            logger.info(f"text_data存在確認: {text_data is not None}")
                            logger.info(f"text_data長さ: {len(text_data) if text_data else 0}")
                            logger.info(f"df_processed行数: {len(df_processed)}")

                            with st.spinner("ファイル保存中..."):
                                # デバッグ情報の表示
                                logger.info("ファイル保存処理開始")

                                # データの存在確認（詳細）
                                if not csv_data:
                                    error_msg = "CSVデータが空です"
                                    st.error(f"❌ {error_msg}")
                                    logger.error(error_msg)

                                    # データ再作成を試行
                                    st.info("📊 データの再作成を試行中...")
                                    try:
                                        logger.info("csv_data再作成開始")
                                        csv_data, text_data = create_download_data(
                                            df_processed,
                                            combine_columns_option,
                                            DATASET_TYPE
                                        )
                                        logger.info(
                                            f"再作成後 - csv_data: {len(csv_data)}, text_data: {len(text_data) if text_data else 0}")

                                        if not csv_data:
                                            st.error("❌ データの再作成にも失敗しました")
                                            logger.error("データ再作成失敗")
                                            return
                                        else:
                                            st.success("✅ データの再作成に成功")
                                            logger.info("データ再作成成功")
                                    except Exception as recreate_error:
                                        st.error(f"❌ データ再作成エラー: {recreate_error}")
                                        logger.error(f"データ再作成エラー: {recreate_error}")
                                        return

                                # データサイズの詳細確認
                                st.info(
                                    f"📊 データサイズ確認: CSV={len(csv_data):,}文字, TXT={len(text_data) if text_data else 0:,}文字")

                                # 保存処理の実行
                                logger.info("save_files_to_output関数呼び出し開始")
                                saved_files = save_files_to_output(
                                    df_processed,
                                    DATASET_TYPE,
                                    csv_data,
                                    text_data
                                )
                                logger.info(f"save_files_to_output関数完了: {saved_files}")

                                if saved_files:
                                    st.success("✅ ファイル保存完了！")
                                    logger.info(f"保存成功: {len(saved_files)}個のファイル")

                                    # 保存されたファイル一覧を表示
                                    with st.expander("📂 保存されたファイル一覧", expanded=True):
                                        for file_type, file_path in saved_files.items():
                                            try:
                                                if Path(file_path).exists():
                                                    file_size = Path(file_path).stat().st_size
                                                    st.write(
                                                        f"**{file_type.upper()}**: `{file_path}` ({file_size:,} bytes) ✅")
                                                    logger.info(
                                                        f"ファイル確認OK: {file_type} - {file_path} ({file_size} bytes)")
                                                else:
                                                    st.write(
                                                        f"**{file_type.upper()}**: `{file_path}` ❌ ファイルが見つかりません")
                                                    logger.error(f"ファイル確認NG: {file_type} - {file_path}")
                                            except Exception as e:
                                                st.write(f"**{file_type.upper()}**: `{file_path}` ❌ エラー: {e}")
                                                logger.error(f"ファイル確認エラー: {file_type} - {e}")

                                        # OUTPUTフォルダの場所を表示
                                        try:
                                            output_path = Path("OUTPUT").resolve()
                                            st.write(f"**保存場所**: `{output_path}`")

                                            # フォルダ内のファイル数
                                            file_count = len(list(output_path.glob("*")))
                                            st.write(f"**フォルダ内ファイル数**: {file_count}個")
                                            logger.info(f"OUTPUTフォルダ: {output_path}, ファイル数: {file_count}")
                                        except Exception as e:
                                            st.error(f"パス情報取得エラー: {e}")
                                            logger.error(f"パス情報エラー: {e}")

                                    # セッション状態の更新
                                    st.session_state['files_saved'] = True
                                    st.session_state['last_save_time'] = datetime.now().isoformat()
                                    logger.info("セッション状態更新完了")
                                else:
                                    st.error("❌ ファイル保存に失敗しました（戻り値が空）")
                                    logger.error("保存関数から空の戻り値")

                        except Exception as e:
                            st.error(f"❌ ファイル保存エラー: {str(e)}")
                            logger.error(f"保存処理総合エラー: {type(e).__name__}: {e}")

                            # トレースバックの詳細ログ
                            import traceback
                            logger.error(f"トレースバック: {traceback.format_exc()}")

                            # デバッグ情報の表示
                            with st.expander("🔧 デバッグ情報", expanded=True):
                                st.code(traceback.format_exc())

                                # 現在のディレクトリ情報
                                try:
                                    import os
                                    current_dir = os.getcwd()
                                    st.write(f"現在のディレクトリ: {current_dir}")

                                    # OUTPUTフォルダの状態確認
                                    output_path = Path("OUTPUT")
                                    if output_path.exists():
                                        st.write(f"OUTPUTフォルダ: 存在 ({output_path.resolve()})")
                                        try:
                                            files = list(output_path.glob("*"))
                                            st.write(f"フォルダ内ファイル数: {len(files)}")
                                            if files:
                                                st.write("ファイル一覧:")
                                                for f in files[:5]:  # 最初の5個を表示
                                                    st.write(f"  - {f.name}")
                                        except Exception as file_error:
                                            st.write(f"フォルダ内容の確認に失敗: {file_error}")
                                    else:
                                        st.write("OUTPUTフォルダ: 存在しない")

                                    # 変数の状態確認
                                    st.write("**変数の状態:**")
                                    st.write(f"- csv_data: {type(csv_data)} (長さ: {len(csv_data) if csv_data else 0})")
                                    st.write(
                                        f"- text_data: {type(text_data)} (長さ: {len(text_data) if text_data else 0})")
                                    st.write(f"- df_processed: {type(df_processed)} (行数: {len(df_processed)})")
                                    st.write(f"- DATASET_TYPE: {DATASET_TYPE}")

                                except Exception as debug_e:
                                    st.write(f"デバッグ情報取得エラー: {debug_e}")
                                    logger.error(f"デバッグ情報エラー: {debug_e}")

                            # 手動でのディレクトリ作成を提案
                            st.info("💡 手動でOUTPUTフォルダを作成してから再試行してください")

                            if st.button("🔄 再試行", key="retry_save"):
                                logger.info("再試行ボタンがクリックされました")
                                st.rerun()

                # 既存の保存済みファイル一覧
                st.write("**📁 既存の保存済みファイル**")
                try:
                    output_dir = Path("OUTPUT")
                    if output_dir.exists() and output_dir.is_dir():
                        saved_files_list = list(output_dir.glob(f"*{DATASET_TYPE}*"))
                        if saved_files_list:
                            with st.expander("保存済みファイル一覧", expanded=False):
                                for file_path in sorted(saved_files_list, reverse=True)[:10]:  # 最新10件
                                    try:
                                        file_stats = file_path.stat()
                                        file_time = datetime.fromtimestamp(file_stats.st_mtime).strftime(
                                            "%Y-%m-%d %H:%M:%S")
                                        st.write(f"- `{file_path.name}` ({file_stats.st_size:,} bytes, {file_time})")
                                    except Exception as e:
                                        st.write(f"- `{file_path.name}` (情報取得エラー: {e})")

                                if len(saved_files_list) > 10:
                                    st.write(f"... 他 {len(saved_files_list) - 10} 個のファイル")
                        else:
                            st.info(f"📂 OUTPUTフォルダに{DATASET_TYPE}関連ファイルは見つかりませんでした")
                    else:
                        st.info("📂 OUTPUTフォルダがまだ作成されていません")

                        # フォルダ作成ボタンを提供
                        if st.button("📁 OUTPUTフォルダを作成", key="create_output_dir"):
                            try:
                                create_output_directory()
                                st.success("✅ OUTPUTフォルダを作成しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ フォルダ作成エラー: {e}")
                except Exception as e:
                    st.error(f"❌ ファイル一覧取得エラー: {e}")
                    logger.error(f"ファイル一覧エラー: {e}")

                # 処理済みデータの保存（セッションステート）
                st.session_state[f'processed_{DATASET_TYPE}'] = {
                    'data'                  : df_processed,
                    'original_rows'         : st.session_state.get('original_rows', len(df)),
                    'processed_rows'        : len(df_processed),
                    'timestamp'             : datetime.now().isoformat(),
                    'files_saved'           : False,  # まだ保存されていない
                    'csv_data_size'         : len(csv_data),
                    'text_data_size'        : len(text_data) if text_data else 0,
                    # 重要：保存用データをセッション状態に保存
                    'csv_data'              : csv_data,
                    'text_data'             : text_data,
                    'combine_columns_option': combine_columns_option
                }

                logger.info(f"医療QAデータ処理完了: {len(df)} → {len(df_processed)}行")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"医療QAデータ処理エラー: {e}")
            if st.checkbox("詳細なエラー情報を表示"):
                st.exception(e)

    # 使用方法の説明
    show_usage_instructions(DATASET_TYPE)

    # セッション状態の表示（デバッグ用）
    if st.sidebar.checkbox("🔧 セッション状態を表示", value=False):
        with st.sidebar.expander("セッション状態", expanded=False):
            if f'processed_{DATASET_TYPE}' in st.session_state:
                processed_info = st.session_state[f'processed_{DATASET_TYPE}']
                st.write(f"**処理済みデータ**: {processed_info['processed_rows']}行")
                st.write(f"**元データ**: {processed_info['original_rows']}行")
                st.write(f"**処理時刻**: {processed_info.get('timestamp', 'N/A')}")

                # ファイル保存状態
                if 'files_saved' in processed_info:
                    if processed_info['files_saved']:
                        st.write("**ファイル保存**: ✅ 完了")
                        if 'last_save_time' in st.session_state:
                            st.write(f"**最終保存時刻**: {st.session_state['last_save_time']}")
                    else:
                        st.write("**ファイル保存**: ❌ 未保存")

                # データサイズ情報
                if 'csv_data_size' in processed_info:
                    st.write(f"**CSVデータサイズ**: {processed_info['csv_data_size']:,} 文字")
                if 'text_data_size' in processed_info:
                    st.write(f"**テキストデータサイズ**: {processed_info['text_data_size']:,} 文字")

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
            else:
                st.write("処理済みデータなし")


# ==================================================
# アプリケーション実行
# ==================================================
if __name__ == "__main__":
    main()

# 実行コマンド:
# streamlit run a30_013_make_rag_data_medical.py --server.port=8503
