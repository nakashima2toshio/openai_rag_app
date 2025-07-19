# python a30_00_rag_dataset_from_huggingface.py
# 下記コードは、ノイズが大きい。 ---> 09_01_rag_
# --------------------------------------------------
# ① カスタマーサポート・FAQデータセット   推奨データセット： Amazon_Polarity
# ② 一般知識・トリビアQAデータセット      推奨データセット： trivia_qa
# ③ 医療質問回答データセット             推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
# ④ 科学・技術QAデータセット             推奨データセット： sciq
# ⑤ 法律・判例QAデータセット             推奨データセット： nguha/legalbench
# --------------------------------------------------
# Embeddingの前処理：　1行1ベクトルになる形が理想
# --------------------------------------------------
import os
import re
import time
import json
from pathlib import Path
from typing import List, Optional
import logging

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

from datasets import load_dataset
import pandas as pd
import tempfile
import textwrap
from pydantic import BaseModel, Field
from tqdm import tqdm

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ヘルパーモジュールをインポート
try:
    from helper_st import (
        UIHelper, MessageManagerUI, ResponseProcessorUI,
        SessionStateManager, error_handler_ui, timer_ui,
        init_page, select_model, InfoPanelManager
    )
    from helper_api import (
        config, logger, TokenManager, OpenAIClient,
        EasyInputMessageParam, ResponseInputTextParam,
        ConfigManager, MessageManager, sanitize_key,
        error_handler, timer
    )
except ImportError as e:
    # Streamlitが利用できない環境での対応
    print(f"ヘルパーモジュールのインポートをスキップ: {e}")
    pass

BASE_DIR = Path(__file__).resolve().parent.parent       # Paslib
THIS_DIR = Path(__file__).resolve().parent              # Paslib

# ----------------------------------------------------
# 1. Customer Support FAQs/ FAQ型のデータ：
# ----------------------------------------------------
def clean_customer_support_faq(csv_path):
    DATASETS_DIR = Path(THIS_DIR) / "datasets"
    csv_path = DATASETS_DIR / "customer_support_faq.csv"
    df = pd.read_csv(csv_path)
    print(df.head(10))

    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(tmp_txt.name, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(textwrap.dedent(f"""\
                Q: {row['question']}
                A: {row['answer']}
            """))
    print("plain-text file:", tmp_txt.name)
    return tmp_txt.name

# 1) 取り出したい構造を Pydantic で宣言 --------------------
class FaqInfo(BaseModel):
    # 抽出対象：answer
    faq_answer: List[str] = Field(..., description="Faq Answer")

# ----------------------------------------------------
# 1. (main)Customer Support FAQs/ FAQ型のデータ：
# ----------------------------------------------------
def customer_support_faq_main():
    # download_dataset()
    # ----------------------------------------------------
    # 1. Customer Support FAQs/ FAQ型のデータ：
    # ----------------------------------------------------
    # csv_path = os.path.join(DATASETS_DIR, "customer_support_faq.csv")
    # print("(1) csv_path:", csv_path)
    #
    # plain_txt_path = set_dataset_to_qa(csv_path)
    # vs_id = create_vector_store_and_upload(plain_txt_path)

    print("(3) Search results:")
    # query_text = "What is your return policy? "
    # query_text = "I want to confirm my order."
    query_text = "I want to check where my shipped package is."
    vs_id = 'vs_684f6202dbe48191bb897b0772563020'
    res_text = standalone_search(query_text, vs_id)
    print("\n\n(3) Search results:-------------------", res_text)

    client = OpenAI()
    model = "gpt-4o-mini"  # モデル名を修正
    messages = f"find and display the answer to the question {query_text} from the following Q: A: information. Information: {res_text}"
    res = client.responses.create(model=model, input=messages)

    print("============")
    for i, txt in enumerate(extract_text_from_response(res), 1):
        print(f"{i}: {txt}\n")

    print("============")
    response = client.responses.parse(model=model, input=messages, text_format=FaqInfo, )
    faq_info: FaqInfo = response.output_parsed
    print(faq_info.model_dump())

# ----------------------------------------------------
# 2. Legal QA — *consumer_contracts_qa
# 列:     Question,Complex_CoT,Response
# 前処理：
# ・「何のために検索・類似度計算するのか」を固め、列ごとの役割を整理すること。
# ・Complex_CoT はそのまま埋め込むとノイズ源になりやすい。要約か除外が無難
# ・医療 QA では用語ゆれ・略語ゆれが検索精度を落としやすいため、正規化詞典をもつと効果大。
# ・実運用前にサンプリングして 近傍検索→人手検証 を行い、前処理の過不足をチェックする
# ----------------------------------------------------
def set_dataset_02(csv_path):
    DATASETS_DIR = Path(THIS_DIR) / "datasets"
    csv_path = DATASETS_DIR / "medical_qa.csv"
    print("(1) csv_path:", csv_path)
    df = pd.read_csv(csv_path)

    # 1. 軽いクリーニング例
    df["Question"] = df["Question"].str.strip()
    df["Response"] = df["Response"].str.strip()

    # 2. CoT 要約 or 除去（ここでは除去）
    df["Clean_CoT"] = df["Complex_CoT"].str.replace(
        r"(?i)^okay,.*?\n", "", regex=True  # 導入句などをざっくり削除
    )

    # 3. 文書生成（ここでは Strategy①）
    df["doc_for_embed"] = df["Question"] + "\n\n" + df["Response"]

    # 4. ベクトル化
    client = OpenAI()
    df["embedding"] = df["doc_for_embed"].apply(
        lambda x: client.embeddings.create(model="text-embedding-3-large", input=x).data[0].embedding
    )

def legal_qa_main():
    csv_path = "medical_qa.csv"
    set_dataset_02(csv_path)

# ----------------------------------------------------
# 3. Medical QA — *medical-o1-reasoning-SFT
#    前処理：
# ----------------------------------------------------
# ========= パス設定を pathlib.Path で統一 =========
# BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
INPUT_CSV: Path = DATASETS_DIR / "medical_qa.csv"
OUTPUT_CSV: Path = DATASETS_DIR / "medical_qa_summarized.csv"
OUTPUT_CLEAN_CSV: Path = DATASETS_DIR / "medical_qa_clean.csv"

# ========= 要約関数 =========
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def summarize_cot(cot_text: str) -> str:
    # 最新 SDK では chat.completions.create を使用
    client = OpenAI()
    # system_content = 'あなたは医療情報の専門家です。以下の医療的な推論内容を簡潔に要約し、診断や判断の要点が明確になるようにしてください。'
    # assistant_content = '医療推論プロセスを、医学的に重要な要点のみを抜粋し100文字以内で要約してください。'
    system_content = 'You are a medical information expert. Please summarize the following medical reasoning content concisely, making the key points of diagnosis and judgment clear.'
    assistant_content = 'Please extract only the medically important key points from the medical reasoning process and summarize them within 100 characters.'
    user_content = cot_text
    # {"role": "system", "content": system_content},
    messages = [
        ChatCompletionSystemMessageParam(role="system", content=system_content),
        ChatCompletionUserMessageParam(role="user", content=user_content),
        ChatCompletionAssistantMessageParam(role="assistant", content=assistant_content),
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",            # モデル名を修正
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def medical_qa_main():
    print("medical_qa_main start ...")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} が見つかりません")

    df = pd.read_csv(INPUT_CSV)
    # 最低限のクリーニング
    df["Question"] = df["Question"].str.strip()
    df["Response"] = df["Response"].str.strip()

    # (A) Complex_CoT 内部の空行を除去 ★追加
    df["Complex_CoT"] = (
        df["Complex_CoT"]
        .astype(str)  # NaN 対策で str 化
        .apply(lambda s: re.sub(r"\n\s*\n+", "\n", s).strip())
    )

    # (B) 行全体が空のレコードを除去（既存ロジック）
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df.dropna(subset=["Question", "Complex_CoT", "Response"], how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # (3) クリーニング後のCSVを保存
    OUTPUT_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CLEAN_CSV, index=False)
    print(f"[OK] cleaned file saved ➜ {OUTPUT_CLEAN_CSV}")

    # Complex_CoT を要約 (rate‑limit 対策で必要なら chunksize/batch ごとに処理)
    # df["CoT"] = df["Complex_CoT"].apply(summarize_cot)
    # tqdm.pandas(desc="Summarizing CoT")
    # df["CoT"] = df["Complex_CoT"].progress_apply(summarize_cot)

    from tqdm import tqdm  # tqdm が未 import の場合は追加
    # 進捗バーを登録（タイトル変更）
    tqdm.pandas(desc="Summarizing CoT (top-100 rows)")

    # ① まず CoT 列を空で作成
    df["CoT"] = pd.NA

    # ② 上から 100 行だけ要約を実行
    df.loc[:99, "CoT"] = df.loc[:99, "Complex_CoT"].progress_apply(summarize_cot)

    # 出力
    print("\ndf_result:------------\n")
    df_result = df[["Question", "CoT", "Response"]]
    print(df_result.head())

    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df_result.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] summarized file saved ➜ {OUTPUT_CSV}")

# ----------------------------------------------------
# INPUT-Docをそのまま使う、embeddingして、vector storeに保存。
# ----------------------------------------------------
INPUT_CLEAN_CSV: Path = DATASETS_DIR / "medical_qa_clean.csv"

def medical_qa_main_short_cut():
    print("medical_qa_main_short_cut start ...")
    if not INPUT_CLEAN_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CLEAN_CSV} が見つかりません")

    df = pd.read_csv(INPUT_CLEAN_CSV)

# ----------------------------------------------------
# medical_qa_search:
#  - vector store へ登録
#    検索
# ----------------------------------------------------
from tempfile import NamedTemporaryFile

def medical_qa_make_vector() -> None:
    print("medical_qa_search start ...")
    # vs_id = 'vs_6852c5fff27c8191a08c0b0b936a1d71'

    # ▼ ① クリーニング済み CSV の存在確認
    if not OUTPUT_CLEAN_CSV.exists():
        raise FileNotFoundError(f"{OUTPUT_CLEAN_CSV} が見つかりません")

    # ▼ ② ファイル読込
    df = pd.read_csv(OUTPUT_CLEAN_CSV)

    # -------------------------------------------------------------
    # (1)  Embedding 用プレーンテキストを一時ファイルに書き出し
    #      形式：  Q: ...\\nA: ...\\n\\n
    # -------------------------------------------------------------
    tmp_txt = NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    up_text = ''
    with tmp_txt as fh:
        for _, row in df.iterrows():
            fh.write(f"Q: {row['Question']}\nA: {row['Response']}\n\n")
            up_text = up_text + f"Q: {row['Question']}\nA: {row['Response']}\n\n"

    # -------------------------------------------------------------
    # (2) Vector Store へアップロード（自前 util を再利用）
    # -------------------------------------------------------------
    upload_name = "medical_store_jp"
    vs_id = create_vector_store_and_upload(up_text, upload_name)

# -------------------------------------------------------------
# (3) RAG 検索テスト（5 問）
#     → Responses API を Retrieval モードで呼ぶ
# -------------------------------------------------------------
def medical_qa_search(vs_id):
    # vs_id = "vs_6852c5fff27c8191a08c0b0b936a1d71"
    questions = [
        "After long-distance travel, a patient presents with swelling and tenderness in the right lower leg and sudden weakness in the left arm and leg. What cardiac abnormality is most likely to explain these findings?",
        "A 5-cm stab wound is located at the upper border of the 8th rib along the left mid-axillary line. Which thoracic structure is most likely to be injured?",
        "In a 61-year-old woman with normal bladder function who complains of stress urinary incontinence, which diagnostic test is the most useful?",
        "A 45-year-old man with a history of chronic heavy alcohol use suddenly develops dysarthria and tremor. What is the most likely diagnosis?",
        "Which disorder, characterized by parkinsonian features and cognitive impairment, is primarily associated with deposition of Lewy bodies?",
    ]

    answer5 = (
        "A1 開存卵円孔（PFO）。深部静脈血栓が右房 → 左房へ短絡し、脳塞栓を起こす paradoxical embolism が機序。\n"
        "A2 左肺下葉（lower lobe of the left lung）。刺創深度と解剖学的位置から血胸・気胸を伴う肺損傷が最も疑われる。\n"
        "A3 膀胱内圧測定（cystometry）。尿道支持機構の障害を評価し、腹圧性尿失禁を確認できる。\n"
        "A4 獲得性肝脳変性（acquired hepatocerebral degeneration）。慢性肝疾患に伴う中枢神経障害で、急性の運動失調・振戦を来す。\n"
        "A5 レビー小体型認知症（Dementia with Lewy bodies）または Lewy 小体パーキンソン病。Lewy 小体の存在が診断の鍵。\n"
        "A1 Patent foramen ovale (PFO). A deep-vein thrombus can shunt from the right atrium to the left atrium, producing a paradoxical embolism that reaches the brain.\n"
        "A2 Lower lobe of the left lung. Given the stab depth and anatomical location, a lung injury with accompanying hemothorax or pneumothorax is most likely.\n"
        "A3 Cystometry (intravesical pressure measurement). Assesses urethral support dysfunction and confirms stress urinary incontinence.\n"
        "A4 Acquired hepatocerebral degeneration. A central-nervous disorder associated with chronic liver disease that causes acute ataxia and tremor.\n"
        "A5 Dementia with Lewy bodies (DLB) or Lewy-body Parkinson's disease. The presence of Lewy bodies is the diagnostic key.\n"
    )

    q1 = "After long-distance travel, a patient presents with swelling and tenderness in the right lower leg and sudden weakness in the left arm and leg. What cardiac abnormality is most likely to explain these findings?"
    a1 = standalone_search(vs_id, q1)

    print("\n======= RAG QA Test =======")
    for i, q in enumerate(questions, 1):
        # Responses エンドポイントでベクターストアを指定
        answer = standalone_search(vs_id, q)
        print(f"\nQ{i}: {q}\nA{i}: {answer}\n{'-'*60}")

    print("\nanswer:=", answer5)

# ----------------------------------------------------
# 4. SciQ — Science MCQ
#    前処理：
# ----------------------------------------------------
def set_dataset_04(csv_path):
    pass

def sci_qa_main():
    pass

# ----------------------------------------------------
# 5. Trivia QA — *rc*（Reading Comprehension）
#    前処理：
# ----------------------------------------------------
def set_dataset_05(csv_path):
    pass

# ----------------------------------------------------
# ① カスタマーサポート・FAQデータセット   推奨データセット： Amazon_Polarity
# ----------------------------------------------------
# データクレンジング処理

def clean_text(text):
    """
    テキストのクレンジング処理
    - 改行の除去
    - 連続した空白を1個の空白にまとめる
    """
    if pd.isna(text):
        return ""

    # 改行の除去
    text = re.sub(r'\n+', ' ', str(text))

    # 連続した空白を1個の空白にまとめる
    text = re.sub(r'\s+', ' ', text)

    # 前後の空白を除去
    text = text.strip()

    return text


def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[
    Optional[List[float]]]:
    """
    OpenAI Embedding APIを使用してテキストリストをバッチでベクトル化

    Args:
        texts: ベクトル化するテキストのリスト
        model: 使用するembeddingモデル (推奨: text-embedding-3-small)
        batch_size: 一度のAPI呼び出しで処理するテキスト数（最大2048）

    Returns:
        埋め込みベクトルのリスト
    """
    client = OpenAI()
    embeddings = []

    # テキストを前処理（改行文字をスペースに置換）
    cleaned_texts = [text.replace("\n", " ") for text in texts]

    logger.info(f"Embedding作成開始: {len(cleaned_texts)}件のテキストを{batch_size}件ずつ処理")

    # バッチごとに処理
    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Embedding作成中"):
        batch = cleaned_texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                input=batch,
                model=model,
                # dimensions=1024  # コスト効率を重視する場合は次元数を削減
            )

            # レスポンスからembeddingを取得
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

            logger.info(
                f"バッチ {i // batch_size + 1}/{(len(cleaned_texts) - 1) // batch_size + 1}: {len(batch)}件処理完了")

            # レート制限対応のための短い待機
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"バッチ {i // batch_size + 1}でエラー: {e}")
            # エラーが発生した場合はNoneで埋める
            embeddings.extend([None] * len(batch))

            # 一時的な問題の場合はリトライ
            time.sleep(2)

    return embeddings


def create_vector_store_from_dataframe(df_clean: pd.DataFrame, store_name: str = "Customer Support FAQ") -> Optional[str]:
    """
    DataFrameからVector Storeを作成（修正版：型エラー対応）

    Args:
        df_clean: クレンジング済みのDataFrame
        store_name: Vector Storeの名前

    Returns:
        Vector Store ID（成功時）またはNone（失敗時）
    """
    client = OpenAI()
    temp_file_path = None
    uploaded_file_id = None

    try:
        # 一時ファイルを作成してアップロード用のJSONLファイルを準備
        # 拡張子を.txtに変更（OpenAI Files APIの制限対応）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            for idx, row in df_clean.iterrows():
                # JSONL形式でデータを書き込み（拡張子は.txtだが中身はJSONL）
                json_line = {
                    "id"  : f"faq_{idx}",
                    "text": row['combined_text']
                }
                temp_file.write(json.dumps(json_line, ensure_ascii=False) + '\n')

            temp_file_path = temp_file.name

        logger.info(f"JSONLファイル作成完了: {temp_file_path}")

        # Step 1: ファイルをOpenAIにアップロード（.txt拡張子でアップロード）
        with open(temp_file_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            uploaded_file_id = uploaded_file.id

        logger.info(f"ファイルアップロード完了: File ID={uploaded_file_id}")

        # Step 2: Vector Storeを作成（問題のあるパラメータを削除）
        vector_store = client.vector_stores.create(
            name=store_name,
            # expires_after パラメータを削除（型エラー対応）
            # chunking_strategy パラメータを削除（型エラー対応）
            metadata={
                "created_by" : "customer_support_faq_processor",
                "version"    : "2025.1",
                "data_format": "jsonl_as_txt"
            }
        )

        logger.info(f"Vector Store作成完了: ID={vector_store.id}")

        # Step 3: Vector StoreにファイルをLinkする（chunking_strategy削除）
        vector_store_file = client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=uploaded_file_id
            # chunking_strategy パラメータを削除（型エラー対応）
        )

        logger.info(f"Vector StoreFileリンク作成: {vector_store_file.id}")

        # Step 4: ファイル処理完了を待機
        max_wait_time = 300  # 最大5分待機
        wait_interval = 5  # 5秒間隔でチェック
        waited_time = 0

        while waited_time < max_wait_time:
            # ファイルステータスを確認
            file_status = client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=uploaded_file_id
            )

            logger.info(f"ファイル処理状況: {file_status.status} (待機時間: {waited_time}秒)")

            if file_status.status == "completed":
                # Vector Store全体のステータスを確認
                updated_vector_store = client.vector_stores.retrieve(vector_store.id)

                logger.info(f"✅ Vector Store作成完了:")
                logger.info(f"  - ID: {vector_store.id}")
                logger.info(f"  - Name: {vector_store.name}")
                logger.info(f"  - ファイル処理状況: {file_status.status}")
                logger.info(f"  - 総ファイル数: {updated_vector_store.file_counts.total}")
                logger.info(f"  - 完了ファイル数: {updated_vector_store.file_counts.completed}")
                logger.info(f"  - 失敗ファイル数: {updated_vector_store.file_counts.failed}")
                logger.info(f"  - ストレージ使用量: {updated_vector_store.usage_bytes} bytes")

                return vector_store.id

            elif file_status.status == "failed":
                logger.error(f"❌ ファイル処理失敗: {file_status.last_error}")
                return None

            elif file_status.status in ["in_progress", "cancelling"]:
                # 処理中の場合は継続して待機
                time.sleep(wait_interval)
                waited_time += wait_interval
            else:
                logger.warning(f"⚠️ 予期しないステータス: {file_status.status}")
                time.sleep(wait_interval)
                waited_time += wait_interval

        # タイムアウトの場合
        logger.error(f"❌ Vector Store作成タイムアウト (制限時間: {max_wait_time}秒)")
        return None

    except Exception as e:
        logger.error(f"Vector Store作成エラー: {e}")
        logger.error(f"エラータイプ: {type(e).__name__}")

        # 具体的なエラー対応の提案
        if "authentication" in str(e).lower():
            logger.error("🔑 APIキーを確認してください。環境変数OPENAI_API_KEYが正しく設定されているか確認。")
        elif "quota" in str(e).lower() or "limit" in str(e).lower():
            logger.error("💳 APIクオータまたはレート制限に達しています。料金プランまたは使用量を確認してください。")
        elif "file" in str(e).lower():
            logger.error("📁 ファイル関連のエラーです。ファイルサイズやフォーマットを確認してください。")
        elif "extension" in str(e).lower() or "format" in str(e).lower():
            logger.error("📄 ファイル拡張子またはフォーマットの問題です。サポートされている形式を確認してください。")

        return None

    finally:
        # 一時ファイルを削除
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("🗑️ 一時ファイルを削除しました")


def validate_embeddings(df_clean: pd.DataFrame) -> bool:
    """
    Embeddingデータの品質を検証

    Args:
        df_clean: Embedding付きのDataFrame

    Returns:
        検証結果（True: 正常, False: 問題あり）
    """
    if 'embedding' not in df_clean.columns:
        logger.error("embedding列が存在しません")
        return False

    null_count = df_clean['embedding'].isnull().sum()
    total_count = len(df_clean)
    success_rate = (total_count - null_count) / total_count * 100

    logger.info(f"Embedding品質検証:")
    logger.info(f"  - 総データ数: {total_count}")
    logger.info(f"  - 成功数: {total_count - null_count}")
    logger.info(f"  - 失敗数: {null_count}")
    logger.info(f"  - 成功率: {success_rate:.1f}%")

    # 成功率が90%未満の場合は警告
    if success_rate < 90:
        logger.warning(f"Embedding成功率が{success_rate:.1f}%と低いです。APIキーやネットワーク接続を確認してください。")
        return False

    return True

# ===
# ----------------------------------------------------
# 1. Customer Support FAQs/ FAQ型のデータ：
# ----------------------------------------------------
def make_vs_id_customer_support_faq():
    logger.info("=== OpenAI API最新版 Vector Store作成処理開始 ===")

    # データファイルの読み込み
    DATASETS_DIR = Path(THIS_DIR) / "datasets"
    csv_path = DATASETS_DIR / "customer_support_faq.csv"

    if not csv_path.exists():
        logger.error(f"データファイルが見つかりません: {csv_path}")
        return

    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)

    logger.info(f"元データ読み込み完了:")
    logger.info(f"  - 行数: {len(df)}")
    logger.info(f"  - 列数: {len(df.columns)}")
    logger.info(f"  - 列名: {list(df.columns)}")

    # df_cleanを新しく定義（クレンジング処理）
    logger.info("データクレンジング開始...")
    df_clean = pd.DataFrame({
        'combined_text': df.apply(
            lambda row: clean_text(str(row['question']) + ' ' + str(row['answer'])),
            axis=1
        )
    })

    logger.info(f"クレンジング後のデータ:")
    logger.info(f"  - 行数: {len(df_clean)}")
    logger.info(f"  - 列数: {len(df_clean.columns)}")
    logger.info(f"  - 列名: {list(df_clean.columns)}")

    # 結果の確認用（最初の3行を表示）
    logger.info("=== 処理結果の確認 ===")
    for i in range(min(3, len(df_clean))):
        logger.info(f"【行 {i + 1}】")
        logger.info(f"  元のquestion: {df.iloc[i]['question'][:100]}...")
        logger.info(f"  元のanswer: {df.iloc[i]['answer'][:100]}...")
        logger.info(f"  連結・クレンジング後: {df_clean.iloc[i]['combined_text'][:200]}...")

    # Embeddingの作成（バッチ処理で効率化）
    logger.info("=== Embedding作成開始 ===")
    logger.info(f"処理対象: {len(df_clean)}件のテキスト")

    # バッチ処理でembeddingを作成
    embeddings = get_embeddings_batch(
        df_clean['combined_text'].tolist(),
        model='text-embedding-3-small',  # コスト効率重視
        batch_size=50  # レート制限を考慮して50件ずつ処理
    )

    # 結果をDataFrameに追加
    df_clean['embedding'] = embeddings

    # Embedding品質の検証
    if not validate_embeddings(df_clean):
        logger.warning("Embeddingの品質に問題があります。処理を続行しますが、結果を確認してください。")

    # 作成したembeddingデータをCSVに保存
    output_dir = Path(THIS_DIR) / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "customer_support_faq_embedded.csv"

    try:
        # Embeddingデータは文字列として保存（CSVの制限対応）
        df_save = df_clean.copy()
        df_save['embedding'] = df_save['embedding'].apply(lambda x: str(x) if x is not None else None)
        df_save.to_csv(output_path, index=False)
        logger.info(f"Embeddingデータを保存: {output_path}")
    except Exception as e:
        logger.error(f"ファイル保存エラー: {e}")

    # Vector Storeの作成
    logger.info("=== Vector Store作成開始 ===")
    vector_store_id = create_vector_store_from_dataframe(df_clean, "Customer Support FAQ v2025")

    if vector_store_id:
        logger.info(f"🎉 Vector Store作成成功!")
        logger.info(f"   Vector Store ID: {vector_store_id}")
        logger.info(f"   このIDを保存して、後でRAG検索に使用してください。")

        # Vector Store IDをファイルに保存
        id_file_path = output_dir / "vector_store_id.txt"
        with open(id_file_path, 'w') as f:
            f.write(vector_store_id)
        logger.info(f"   Vector Store IDを保存: {id_file_path}")

    else:
        logger.error("❌ Vector Storeの作成に失敗しました。")
        logger.error("   エラーログを確認し、APIキーやネットワーク接続を確認してください。")

    # 処理完了サマリー
    logger.info("=== 処理完了サマリー ===")
    logger.info(f"✅ 処理済みデータ: {len(df_clean)}件")
    logger.info(f"✅ 成功したEmbedding: {df_clean['embedding'].notna().sum()}件")
    logger.info(f"✅ Vector Store: {'作成成功' if vector_store_id else '作成失敗'}")

    # コスト推定
    total_tokens = sum(len(text.split()) * 1.3 for text in df_clean['combined_text'])  # 概算
    estimated_cost = total_tokens * 0.00002 / 1000  # text-embedding-3-small料金
    logger.info(f"💰 推定コスト: 約${estimated_cost:.4f} (概算)")

# 7-16-1: Vector Store ID: vs_68775be00d84819192ecc2b9c1039b89

# ----------------------------------------------------
# 2. Legal QA — *consumer_contracts_qa
# 列: Question,Complex_CoT,Response
# ----------------------------------------------------

# 不足している関数を追加（ダミー実装）
def standalone_search(vs_id, query):
    """検索関数のダミー実装"""
    return f"検索結果: {query} (Vector Store: {vs_id})"

def extract_text_from_response(response):
    """レスポンステキスト抽出のダミー実装"""
    return ["サンプルレスポンス"]

def create_vector_store_and_upload(text, name):
    """Vector Store作成のダミー実装"""
    return "vs_sample_id"

def main():
    """メイン関数"""
    print("RAGデータセット処理プログラム")
    print("エラー修正済み版")

if __name__ == "__main__":
    main()
