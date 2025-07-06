# rag_dataset_processor.py
# --------------------------------------------------
# HuggingFaceデータセットを使用したRAGシステム
# 5つのデータセットの処理・埋め込み・検索を統合管理
# --------------------------------------------------

import os
import re
import time
import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)


# ==================================================
# 設定とデータクラス
# ==================================================
@dataclass
class DatasetConfig:
    # データセット設定
    name: str
    hf_path: str
    config: Optional[str]
    split: str
    description: str


class DatasetType(Enum):
    # データセットタイプ
    CUSTOMER_FAQ = "customer_support_faq"
    TRIVIA_QA = "trivia_qa"
    MEDICAL_QA = "medical_qa"
    SCIENCE_QA = "sciq_qa"
    LEGAL_QA = "legal_qa"


# データセット設定
DATASET_CONFIGS = {
    DatasetType.CUSTOMER_FAQ: DatasetConfig(
        name="customer_support_faq",
        hf_path="MakTek/Customer_support_faqs_dataset",
        config=None,
        split="train",
        description="カスタマーサポートFAQデータセット"
    ),
    DatasetType.TRIVIA_QA   : DatasetConfig(
        name="trivia_qa",
        hf_path="trivia_qa",
        config="rc",
        split="train",
        description="一般知識・トリビアQAデータセット"
    ),
    DatasetType.MEDICAL_QA  : DatasetConfig(
        name="medical_qa",
        hf_path="FreedomIntelligence/medical-o1-reasoning-SFT",
        config="en",
        split="train",
        description="医療質問回答データセット"
    ),
    DatasetType.SCIENCE_QA  : DatasetConfig(
        name="sciq_qa",
        hf_path="sciq",
        config=None,
        split="train",
        description="科学・技術QAデータセット"
    ),
    DatasetType.LEGAL_QA    : DatasetConfig(
        name="legal_qa",
        hf_path="nguha/legalbench",
        config="consumer_contracts_qa",
        split="train",
        description="法律・判例QAデータセット"
    )
}


# ==================================================
# ベースクラス
# ==================================================
class BaseDatasetProcessor(ABC):
    # データセット処理の基底クラス# 

    def __init__(self, dataset_type: DatasetType, base_dir: Path = None):
        self.dataset_type = dataset_type
        self.config = DATASET_CONFIGS[dataset_type]
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        self.client = OpenAI()
        self.vector_store_id: Optional[str] = None

    @property
    def csv_path(self) -> Path:
        # CSVファイルパス# 
        return self.datasets_dir / f"{self.config.name}.csv"

    @property
    def clean_csv_path(self) -> Path:
        # クリーニング済みCSVファイルパス# 
        return self.datasets_dir / f"{self.config.name}_clean.csv"

    @property
    def processed_csv_path(self) -> Path:
        # 処理済みCSVファイルパス# 
        return self.datasets_dir / f"{self.config.name}_processed.csv"

    def download_dataset(self) -> pd.DataFrame:
        # データセットのダウンロード# 
        print(f"▼ Downloading {self.config.name}...")

        if self.csv_path.exists():
            print(f"  Using cached CSV: {self.csv_path}")
            return pd.read_csv(self.csv_path)

        ds = load_dataset(
            path=self.config.hf_path,
            name=self.config.config,
            split=self.config.split,
        )

        # CSV形式で保存
        df = ds.to_pandas()
        df.to_csv(self.csv_path, index=False)
        print(f"  Saved CSV → {self.csv_path}")

        return df

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # データの前処理（サブクラスで実装）# 
        pass

    @abstractmethod
    def create_embedding_text(self, row: pd.Series) -> str:
        # 埋め込み用テキストの作成（サブクラスで実装）# 
        pass

    def create_vector_store(self, df: pd.DataFrame) -> str:
        # ベクトルストアの作成# 
        print(f"\n▼ Creating vector store for {self.config.name}...")

        # 一時ファイルに埋め込み用テキストを書き出し
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating embeddings"):
                embedding_text = self.create_embedding_text(row)
                tmp.write(embedding_text + "\n\n")
            tmp_path = tmp.name

        # ベクトルストアの作成
        vs = self.client.vector_stores.create(name=f"{self.config.name}_store")
        self.vector_store_id = vs.id

        # ファイルのアップロード
        with open(tmp_path, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose="assistants")

        # ベクトルストアにファイルを追加
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file_obj.id
        )

        # インデックス完了を待機
        print("  Waiting for indexing to complete...")
        while self.client.vector_stores.retrieve(self.vector_store_id).status != "completed":
            time.sleep(2)

        # 一時ファイルを削除
        os.unlink(tmp_path)

        print(f"  Vector store created: {self.vector_store_id}")
        return self.vector_store_id

    def search(self, query: str, max_results: int = 1) -> List[Dict[str, Any]]:
        # ベクトルストアでの検索# 
        if not self.vector_store_id:
            raise ValueError("Vector store not created yet")

        results = self.client.vector_stores.search(
            vector_store_id=self.vector_store_id,
            query=query,
            max_results=max_results
        )

        return [
            {
                "score"  : r.score,
                "content": r.content[0].text.strip() if r.content else "",
            }
            for r in results.data
        ]

    def process_pipeline(self) -> str:
        # 完全な処理パイプライン# 
        # 1. ダウンロード
        df = self.download_dataset()

        # 2. 前処理
        df_processed = self.preprocess(df)

        # 3. 保存
        df_processed.to_csv(self.processed_csv_path, index=False)
        print(f"  Processed data saved → {self.processed_csv_path}")

        # 4. ベクトルストア作成
        vs_id = self.create_vector_store(df_processed)

        return vs_id


# ==================================================
# 各データセット用のプロセッサー
# ==================================================
class CustomerFAQProcessor(BaseDatasetProcessor):
    # カスタマーサポートFAQプロセッサー# 

    class FAQAnswer(BaseModel):
        # FAQ回答の構造化出力# 
        answer: str = Field(..., description="FAQ Answer")
        confidence: float = Field(..., description="Confidence score")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 前処理: クリーニングのみ# 
        df = df.copy()
        df['question'] = df['question'].str.strip()
        df['answer'] = df['answer'].str.strip()

        # 空行を削除
        df = df.dropna(subset=['question', 'answer'])

        return df

    def create_embedding_text(self, row: pd.Series) -> str:
        # Q&A形式でテキスト作成# 
        return f"Q: {row['question']}\nA: {row['answer']}"

    def extract_answer(self, query: str, search_results: List[Dict[str, Any]]) -> FAQAnswer:
        # 検索結果から回答を抽出# 
        if not search_results:
            return self.FAQAnswer(answer="No relevant answer found", confidence=0.0)

        # 最も関連性の高い結果を使用
        best_result = search_results[0]

        # Responses APIで構造化出力
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="Extract the answer from the FAQ information."
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Query: {query}\n\nFAQ Information:\n{best_result['content']}"
            )
        ]

        response = self.client.responses.parse(
            model="gpt-4o-mini",
            input=messages,
            text_format=self.FAQAnswer
        )

        return response.output_parsed


class MedicalQAProcessor(BaseDatasetProcessor):
    # 医療QAプロセッサー# 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_cot_length = 200  # CoTの最大長

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def summarize_cot(self, cot_text: str) -> str:
        # Complex CoTの要約# 
        if not cot_text or len(cot_text) < 50:
            return cot_text

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="You are a medical expert. Summarize the medical reasoning concisely."
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Summarize this medical reasoning in {self.max_cot_length} characters:\n{cot_text}"
            )
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=self.max_cot_length
        )

        return response.choices[0].message.content.strip()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 前処理: CoTの要約とクリーニング# 
        df = df.copy()

        # 基本的なクリーニング
        for col in ['Question', 'Response', 'Complex_CoT']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Complex_CoTの空行除去
        df['Complex_CoT'] = df['Complex_CoT'].apply(
            lambda s: re.sub(r'\n\s*\n+', '\n', s).strip()
        )

        # 空行を削除
        df = df.dropna(subset=['Question', 'Response'], how='all')
        df = df.reset_index(drop=True)

        # CoTの要約（上位100行のみ、処理時間短縮のため）
        print("Summarizing Complex CoT...")
        if len(df) > 100:
            print(f"  Processing top 100 rows out of {len(df)}")
            df_top = df.head(100).copy()
            tqdm.pandas(desc="Summarizing CoT")
            df_top['CoT_Summary'] = df_top['Complex_CoT'].progress_apply(self.summarize_cot)
            df.loc[:99, 'CoT_Summary'] = df_top['CoT_Summary']
            df.loc[100:, 'CoT_Summary'] = ''
        else:
            tqdm.pandas(desc="Summarizing CoT")
            df['CoT_Summary'] = df['Complex_CoT'].progress_apply(self.summarize_cot)

        return df

    def create_embedding_text(self, row: pd.Series) -> str:
        # 医療QA用の埋め込みテキスト# 
        cot_text = row.get('CoT_Summary', '')
        if cot_text:
            return f"Q: {row['Question']}\nReasoning: {cot_text}\nA: {row['Response']}"
        else:
            return f"Q: {row['Question']}\nA: {row['Response']}"


class TriviaQAProcessor(BaseDatasetProcessor):
    # トリビアQAプロセッサー# 

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 前処理: 複数回答の統合# 
        df = df.copy()

        # questionとanswerの抽出
        if 'question' in df.columns:
            df['question_text'] = df['question']

        # answer列の処理（複数回答の場合）
        if 'answer' in df.columns:
            df['answer_text'] = df['answer'].apply(
                lambda x: x['value'] if isinstance(x, dict) else str(x)
            )

        # 必要な列のみ保持
        keep_cols = ['question_text', 'answer_text']
        df = df[[col for col in keep_cols if col in df.columns]]

        return df

    def create_embedding_text(self, row: pd.Series) -> str:
        # トリビア用の埋め込みテキスト# 
        return f"Q: {row.get('question_text', '')}\nA: {row.get('answer_text', '')}"


class ScienceQAProcessor(BaseDatasetProcessor):
    # 科学QAプロセッサー# 

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 前処理: 選択肢の整形# 
        df = df.copy()

        # 選択肢を統合
        if 'distractor1' in df.columns:
            df['choices'] = df.apply(
                lambda row: [
                    row.get('correct_answer', ''),
                    row.get('distractor1', ''),
                    row.get('distractor2', ''),
                    row.get('distractor3', '')
                ],
                axis=1
            )

        return df

    def create_embedding_text(self, row: pd.Series) -> str:
        # 科学QA用の埋め込みテキスト# 
        question = row.get('question', '')
        answer = row.get('correct_answer', '')
        support = row.get('support', '')

        text = f"Q: {question}\nA: {answer}"
        if support:
            text += f"\nExplanation: {support}"

        return text


class LegalQAProcessor(BaseDatasetProcessor):
    # 法律QAプロセッサー# 

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 前処理: 法律用語の正規化# 
        df = df.copy()

        # 列名の確認と調整
        if 'question' in df.columns:
            df['Question'] = df['question']
        if 'answer' in df.columns:
            df['Answer'] = df['answer']

        # テキストのクリーニング
        for col in ['Question', 'Answer']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # 法律用語の正規化（例）
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

        return df

    def create_embedding_text(self, row: pd.Series) -> str:
        # 法律QA用の埋め込みテキスト# 
        return f"Legal Q: {row.get('Question', '')}\nLegal A: {row.get('Answer', '')}"


# ==================================================
# 統合マネージャー
# ==================================================
class RAGDatasetManager:
    # RAGデータセット統合管理クラス# 

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.processors = {
            DatasetType.CUSTOMER_FAQ: CustomerFAQProcessor(DatasetType.CUSTOMER_FAQ, base_dir),
            DatasetType.MEDICAL_QA  : MedicalQAProcessor(DatasetType.MEDICAL_QA, base_dir),
            DatasetType.TRIVIA_QA   : TriviaQAProcessor(DatasetType.TRIVIA_QA, base_dir),
            DatasetType.SCIENCE_QA  : ScienceQAProcessor(DatasetType.SCIENCE_QA, base_dir),
            DatasetType.LEGAL_QA    : LegalQAProcessor(DatasetType.LEGAL_QA, base_dir),
        }
        self.vector_store_ids = {}

    def process_dataset(self, dataset_type: DatasetType) -> str:
        # 特定のデータセットを処理# 
        processor = self.processors[dataset_type]
        vs_id = processor.process_pipeline()
        self.vector_store_ids[dataset_type] = vs_id
        return vs_id

    def process_all_datasets(self) -> Dict[DatasetType, str]:
        # 全データセットを処理# 
        for dataset_type in DatasetType:
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing {dataset_type.value}")
                print('=' * 60)
                self.process_dataset(dataset_type)
            except Exception as e:
                print(f"Error processing {dataset_type.value}: {e}")

        return self.vector_store_ids

    def test_search(self, dataset_type: DatasetType, queries: List[str]) -> None:
        # 検索テスト# 
        processor = self.processors[dataset_type]

        if dataset_type not in self.vector_store_ids:
            print(f"Vector store for {dataset_type.value} not found")
            return

        processor.vector_store_id = self.vector_store_ids[dataset_type]

        print(f"\n▼ Testing search for {dataset_type.value}")
        print("=" * 60)

        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 40)

            results = processor.search(query, max_results=3)

            if isinstance(processor, CustomerFAQProcessor):
                # 構造化出力を使用
                answer = processor.extract_answer(query, results)
                print(f"Answer: {answer.answer}")
                print(f"Confidence: {answer.confidence:.2f}")
            else:
                # 通常の検索結果表示
                for j, result in enumerate(results, 1):
                    print(f"\nResult {j} (Score: {result['score']:.3f}):")
                    print(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'])

    def save_vector_store_mapping(self) -> None:
        # ベクトルストアIDのマッピングを保存# 
        mapping_file = self.base_dir / "vector_store_mapping.json"
        mapping = {
            dt.value: vs_id
            for dt, vs_id in self.vector_store_ids.items()
        }

        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"\nVector store mapping saved to: {mapping_file}")

    def load_vector_store_mapping(self) -> None:
        # ベクトルストアIDのマッピングを読み込み# 
        mapping_file = self.base_dir / "vector_store_mapping.json"

        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)

            self.vector_store_ids = {
                DatasetType(name): vs_id
                for name, vs_id in mapping.items()
            }

            # プロセッサーにもセット
            for dt, vs_id in self.vector_store_ids.items():
                self.processors[dt].vector_store_id = vs_id

            print(f"Vector store mapping loaded from: {mapping_file}")


# ==================================================
# メイン実行
# ==================================================
def main():
    # メイン処理# 
    manager = RAGDatasetManager()

    # 既存のベクトルストアマッピングを読み込み
    manager.load_vector_store_mapping()

    # 処理モードの選択
    print("Select mode:")
    print("1. Process all datasets")
    print("2. Process specific dataset")
    print("3. Test search only")

    mode = input("Enter mode (1-3): ").strip()

    if mode == "1":
        # 全データセット処理
        manager.process_all_datasets()
        manager.save_vector_store_mapping()

    elif mode == "2":
        # 特定データセット処理
        print("\nAvailable datasets:")
        for i, dt in enumerate(DatasetType, 1):
            print(f"{i}. {dt.value}")

        idx = int(input("Select dataset (1-5): ")) - 1
        dataset_type = list(DatasetType)[idx]

        vs_id = manager.process_dataset(dataset_type)
        manager.save_vector_store_mapping()
        print(f"\nVector store ID: {vs_id}")

    elif mode == "3":
        # 検索テストのみ
        if not manager.vector_store_ids:
            print("No vector stores found. Please process datasets first.")
            return

    # 検索テスト
    print("\n" + "=" * 60)
    print("SEARCH TESTS")
    print("=" * 60)

    # テストクエリ
    test_queries = {
        DatasetType.CUSTOMER_FAQ: [
            "What is your return policy?",
            "I want to check where my shipped package is.",
            "How can I reset my password?"
        ],
        DatasetType.MEDICAL_QA  : [
            "What cardiac abnormality explains leg swelling after travel with sudden arm weakness?",
            "Which diagnostic test is best for stress urinary incontinence?",
            "What disorder shows parkinsonian features with Lewy bodies?"
        ],
        DatasetType.TRIVIA_QA   : [
            "Who invented the telephone?",
            "What is the capital of France?",
            "When was the United Nations founded?"
        ],
        DatasetType.SCIENCE_QA  : [
            "What is photosynthesis?",
            "How do vaccines work?",
            "What causes earthquakes?"
        ],
        DatasetType.LEGAL_QA    : [
            "What is a breach of contract?",
            "What are consumer rights?",
            "What is intellectual property?"
        ]
    }

    # 各データセットでテスト
    for dataset_type in DatasetType:
        if dataset_type in manager.vector_store_ids:
            manager.test_search(dataset_type, test_queries[dataset_type])


if __name__ == "__main__":
    main()
