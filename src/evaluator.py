import os
import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

def evaluate_hit_mrr(vectorstore, test_data, k_values=[1, 3, 5, 10], reranker=None):
    """Evaluate with hit@k and MRR."""
    hit_k = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for query, correct_url in tqdm.tqdm(test_data, desc="Evaluating queries"):
        top_k = max(k_values)
        retrieved_docs = vectorstore.similarity_search(query, k=top_k)
        retrieved_urls = [doc.metadata.get('source', '') for doc in retrieved_docs]

        if reranker is not None:
            pairs = [[query, doc.page_content] for doc in retrieved_docs]
            scores = reranker.predict(pairs)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            retrieved_urls = [retrieved_urls[i] for i in sorted_indices]

        found_rank = None
        for idx, url in enumerate(retrieved_urls):
            if url == correct_url:
                found_rank = idx + 1
                break

        if found_rank is not None:
            for k in k_values:
                if found_rank <= k:
                    hit_k[k] += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    total_queries = len(test_data)
    print(f"\nEvaluation on {total_queries} queries:")
    for k in k_values:
        hit_rate = hit_k[k] / total_queries
        print(f"Hit@{k}: {hit_rate:.3f}")
    mrr = sum(reciprocal_ranks) / total_queries
    print(f"MRR: {mrr:.3f}")

def load_test_data(filepath):
    """Load test data from file."""
    test_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(':', 1)
            if len(parts) == 2:
                question = parts[0].strip()
                url = parts[1].strip()
                test_data.append((question, url))
    return test_data

def main():
    VECTORSTORE_DIR = os.path.join("vectorstores", "my_articles_store", "faiss_index")
    TEST_DATA_FILE = "test_data.txt"

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    test_data = load_test_data(TEST_DATA_FILE)

    # Evaluate pure FAISS retrieval
    print("🔍 Evaluating pure FAISS retrieval:")
    evaluate_hit_mrr(vectorstore, test_data)

    # Evaluate with CrossEncoder reranker
    print("\n🔍 Evaluating with CrossEncoder reranker:")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    evaluate_hit_mrr(vectorstore, test_data, reranker=reranker)

if __name__ == "__main__":
    main()
