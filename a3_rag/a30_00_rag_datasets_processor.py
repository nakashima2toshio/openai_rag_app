# a30_00_rag_datasets_processor.py
# test_rag_system.py
# RAGシステムのテストスクリプト

from pathlib import Path
from a30_02_datasets_to_rag_search import RAGDatasetManager, DatasetType

def test_customer_faq():
    # カスタマーFAQのテスト# 
    print("\n" + "=" * 60)
    print("Testing Customer FAQ Dataset")
    print("=" * 60)

    manager = RAGDatasetManager()

    # 既存のVector Storeがあれば使用
    manager.load_vector_store_mapping()

    if DatasetType.CUSTOMER_FAQ not in manager.vector_store_ids:
        print("Processing Customer FAQ dataset...")
        manager.process_dataset(DatasetType.CUSTOMER_FAQ)
        manager.save_vector_store_mapping()

    # テストクエリ
    test_queries = [
        "What is your return policy?",
        "How can I track my order?",
        "I want to reset my password",
        "What payment methods do you accept?",
        "How do I contact customer support?"
    ]

    manager.test_search(DatasetType.CUSTOMER_FAQ, test_queries)


def test_medical_qa():
    # 医療QAのテスト# 
    print("\n" + "=" * 60)
    print("Testing Medical QA Dataset")
    print("=" * 60)

    manager = RAGDatasetManager()
    manager.load_vector_store_mapping()

    if DatasetType.MEDICAL_QA not in manager.vector_store_ids:
        print("Processing Medical QA dataset...")
        manager.process_dataset(DatasetType.MEDICAL_QA)
        manager.save_vector_store_mapping()

    # 医療関連のテストクエリ
    test_queries = [
        "Patient with leg swelling after travel and sudden arm weakness - what cardiac issue?",
        "Best diagnostic test for stress urinary incontinence in 61-year-old woman?",
        "Chronic alcohol use with sudden tremor and dysarthria - diagnosis?",
        "Parkinsonian symptoms with Lewy bodies - which disorder?",
        "Stab wound at 8th rib mid-axillary line - what structure injured?"
    ]

    manager.test_search(DatasetType.MEDICAL_QA, test_queries)


def test_quick_search():
    # 既存のVector Storeで素早く検索テスト# 
    print("\n" + "=" * 60)
    print("Quick Search Test")
    print("=" * 60)

    manager = RAGDatasetManager()
    manager.load_vector_store_mapping()

    if not manager.vector_store_ids:
        print("No vector stores found. Please run full test first.")
        return

    # 各データセットで1つずつクエリをテスト
    quick_queries = {
        DatasetType.CUSTOMER_FAQ: ["What is your return policy?"],
        DatasetType.MEDICAL_QA  : ["What causes Lewy body dementia?"],
        DatasetType.TRIVIA_QA   : ["Who invented the telephone?"],
        DatasetType.SCIENCE_QA  : ["What is photosynthesis?"],
        DatasetType.LEGAL_QA    : ["What is breach of contract?"]
    }

    for dataset_type, queries in quick_queries.items():
        if dataset_type in manager.vector_store_ids:
            print(f"\n--- {dataset_type.value} ---")
            manager.test_search(dataset_type, queries)


def inspect_vector_stores():
    # Vector Storeの状態を確認# 
    from helper import inspect_vector_store

    print("\n" + "=" * 60)
    print("Vector Store Information")
    print("=" * 60)

    manager = RAGDatasetManager()
    manager.load_vector_store_mapping()

    for dataset_type, vs_id in manager.vector_store_ids.items():
        print(f"\n{dataset_type.value}:")
        info = inspect_vector_store(vs_id)
        for key, value in info.items():
            print(f"  {key}: {value}")


def main():
    # メインテスト実行# 
    print("RAG System Test Suite")
    print("====================")
    print("1. Test Customer FAQ only")
    print("2. Test Medical QA only")
    print("3. Quick search test (all datasets)")
    print("4. Inspect vector stores")
    print("5. Full test (all datasets)")

    choice = input("\nSelect test option (1-5): ").strip()

    if choice == "1":
        test_customer_faq()
    elif choice == "2":
        test_medical_qa()
    elif choice == "3":
        test_quick_search()
    elif choice == "4":
        inspect_vector_stores()
    elif choice == "5":
        # フルテスト
        manager = RAGDatasetManager()
        manager.process_all_datasets()
        manager.save_vector_store_mapping()

        # 全データセットでテスト
        test_quick_search()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
