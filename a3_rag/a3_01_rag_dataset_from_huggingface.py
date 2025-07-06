# python a3_01_rag_dataset_from_huggingface.py
# 下記コードは、ノイズが大きい。 ---> 09_01_rag_
# --------------------------------------------------
# ① カスタマーサポート・FAQデータセット   推奨データセット： Amazon_Polarity
# ② 一般知識・トリビアQAデータセット      推奨データセット： trivia_qa
# ③ 医療質問回答データセット             推奨データセット： FreedomIntelligence/medical-o1-reasoning-SFT
# ④ 科学・技術QAデータセット             推奨データセット： sciq
# ⑤ 法律・判例QAデータセット             推奨データセット： nguha/legalbench
# --------------------------------------------------
# 1. いま得られている結果をどう評価するか？
# - 多くの RAG ワークフローでは 要点抽出または除外 が推奨です。
#    CoT を保持しておきたい場合は “raw_cot” を別フィールドとしてストレージに保存し、
#    検索ヒット後に参照すると良いでしょう。
# --------------------------------------------------
# 観点	            評価
# 検索自体のヒット	    上位 0.746 のチャンク内に「返品ポリシーを教えてください。」
#                   →30 日以内で全額返金という正しい回答が含まれており、リコール（再現率）は OK。
# 精度 (Precision)	返ってきたチャンクが “Q/A を 10 問以上まとめて 1 塊” になっているため、
#                   ノイズが多く余計な QA も一緒に返っている。
# スコア分布	        0.746 → 0.676 → 0.636 ときれいに降下しており、ベクトル検索自体は機能している。
# 次の課題	        ・チャンクが大き過ぎる
#                   ・回答文だけ抽出してユーザーに返すロジックが無い
# --------------------------------------------------
import os
import re
import time
from pathlib import Path
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from openai.types.vector_store_create_params import ExpiresAfter  # ★追加

from datasets import load_dataset
import pandas as pd, tempfile, textwrap
from pydantic import BaseModel, Field
from tqdm import tqdm

from a0_common_helper.helper import (
    # init_page,
    # init_messages,
    # select_model,
    # sanitize_key,
    # get_default_messages,
    create_vector_store_and_upload,
    standalone_search,
    extract_text_from_response,
)

BASE_DIR = Path(__file__).resolve().parent.parent       # Paslib
THIS_DIR = Path(__file__).resolve().parent              # Paslib
HELPER_DIR = os.path.join(BASE_DIR, 'a0_common_helper') # string

datasets_to_download = [
    {
        "name": "customer_support_faq",
        "hfpath": "MakTek/Customer_support_faqs_dataset",
        "config": None,
        "split": "train",
    },
    {
        "name": "trivia_qa",
        "hfpath": "trivia_qa",
        "config": "rc",
        "split": "train",
    },
    {
        "name": "medical_qa",
        "hfpath": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "config": "en",
        "split": "train",
    },
    {
        "name": "sciq_qa",
        "hfpath": "sciq",
        "config": None,
        "split": "train",
    },
    {
        "name": "legal_qa",
        "hfpath": "nguha/legalbench",
        "config": "consumer_contracts_qa",  # ★必須
        "split": "train",
    },
]


def download_dataset():
    DATA_DIR = Path("datasets")
    DATA_DIR.mkdir(exist_ok=True)

    for d in datasets_to_download:
        print(f"▼ downloading {d['name']} …")
        ds = load_dataset(
            path=d["hfpath"],
            name=d["config"],
            split=d["split"],
        )

        # # Arrow 形式 → data/<name>
        # arrow_path = DATA_DIR / d["name"]
        # ds.save_to_disk(arrow_path)
        # print(f"  saved dataset ➜ {arrow_path}")

        # CSV 形式 → data/<name>.csv
        csv_path = DATA_DIR / f"{d['name']}.csv"
        ds.to_pandas().to_csv(csv_path, index=False)
        print(f"  saved CSV     ➜ {csv_path}")

    print("\n[OK] All datasets downloaded & saved.")

# ----------------------------------------------------
# 1. Customer Support FAQs/ FAQ型のデータ：
#    前処理：「Q: … A: …」形式へ変換
# ----------------------------------------------------
def set_dataset_to_qa(csv_path):
    # 「Q: … A: …」形式へ変換し一時ファイルに書き出し
    df = pd.read_csv(csv_path)
    print(df.head())

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
    model = "gpt-4.1-mini"
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
    csv_path = os.path.join(DATASETS_DIR, "medical_qa.csv")
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
BASE_DIR = Path(__file__).resolve().parent
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
        model="gpt-4.1-nano",            # GPT-4 系モデルに変更可
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
# [QAの方針]
# 質問は診断・病態・検査選択など医療現場で有用な内容に重点を置く。
# 回答は原則として Response 列を優先し、必要に応じて CoT 要約も参考に自然かつ端的にまとめる。
# 質問・回答ともに英語で出力し、日本語が含まれる場合は自然な英文に変換する。
# ----------------------------------------------------
#   Question(Q) Recommended Answer(A) (Key points that RAG should return)
# 1 (Q) What is the most likely cardiac abnormality that could explain the right lower leg swelling and tenderness after long-distance travel, along with sudden weakness in the left upper and lower limbs?
#   (A) Patent foramen ovale (PFO). The mechanism is paradoxical embolism where deep vein thrombosis causes a right-to-left atrial shunt, leading to cerebral embolism.
# 2 (Q) What chest structure is most likely to be injured in a case of a 5 cm stab wound at the upper border of the left 8th rib along the left midaxillary line?
#   (A) Lower lobe of the left lung. Based on the depth of penetration and anatomical location, lung injury with hemothorax and pneumothorax is most suspected.
# 3 (Q) What is the most useful diagnostic test for a 61-year-old woman with normal bladder function who complains of stress urinary incontinence?
#   (A) Cystometry. This can evaluate dysfunction of the urethral support mechanism and confirm stress urinary incontinence.
# 4 (Q) What is the most likely diagnosis for a 45-year-old man with a history of chronic heavy alcohol consumption who suddenly develops dysarthria and tremor?
#   (A) Acquired hepatocerebral degeneration. This is a central nervous system disorder associated with chronic liver disease that causes acute ataxia and tremor.
# 5 (Q) What disease shows Parkinsonian symptoms and cognitive impairment, with Lewy body deposition being central to its pathology?
#   (A) Dementia with Lewy bodies or Lewy body Parkinson's disease. The presence of Lewy bodies is the key to diagnosis.
# ----------------------------------------------------
#	質問(Q) 推奨される回答(A) （RAG が返すべき要点）
# 1	(Q) 長距離移動後に右下腿の腫脹・圧痛があり、左上肢・左下肢に突然の脱力が出現した患者で、これらを説明し得る最も可能性の高い心臓の異常は何か？
#   (A) 開存卵円孔（PFO）。深部静脈血栓が右房 → 左房へ短絡し、脳塞栓を起こす paradoxical embolism が機序。
# 2	(Q) 左第 8 肋骨上縁・左中腋窩線に 5 cm の刺創がある場合、損傷されている可能性が最も高い胸部構造はどこか？
#   (A) 左肺下葉（lower lobe of the left lung）。刺創深度と解剖学的位置から血胸・気胸を伴う肺損傷が最も疑われる。
# 3	(Q) 膀胱機能が正常で腹圧性尿失禁を訴える 61 歳女性において、診断に最も有用な検査は何か？
#   (A) 膀胱内圧測定（cystometry）。尿道支持機構の障害を評価し、腹圧性尿失禁を確認できる。
# 4	(Q) 慢性アルコール多飲歴の 45 歳男性が突然の構音障害と振戦を呈した場合、最も考えられる診断は何か？
#   (A) 獲得性肝脳変性（acquired hepatocerebral degeneration）。慢性肝疾患に伴う中枢神経障害で、急性の運動失調・振戦を来す。
# 5	(Q) パーキンソン様症状と認知障害を示し、Lewy 小体の沈着が病態の中心となる疾患は何か？
#   (A) レビー小体型認知症（Dementia with Lewy bodies）または Lewy 小体パーキンソン病。Lewy 小体の存在が診断の鍵。
# ----------------------------------------------------
from tempfile import NamedTemporaryFile
# from tqdm import tqdm

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
        "A5 Dementia with Lewy bodies (DLB) or Lewy-body Parkinson’s disease. The presence of Lewy bodies is the diagnostic key.\n"
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
def main():
    # customer_support_faq_main()
    # legal_qa_main()
    # medical_qa_main()
    # medical_qa_make_vector()
    medical_qa_search()

if __name__ == "__main__":
    main()
