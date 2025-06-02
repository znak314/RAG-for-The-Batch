import streamlit as st
import openai
import torch
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# ==== Load environment ====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Load FAISS vectorstore ====
VECTORSTORE_DIR = os.path.join("vectorstores", "my_articles_store", "faiss_index")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==== Streamlit UI ====
st.set_page_config(page_title="Multimodal RAG Search + GPT-4o", page_icon="📰", layout="wide")
st.title("🔍 Multimodal RAG Search + GPT-4o")
st.write("Enter a query to search relevant articles (with images) and generate an answer using GPT-4o.")
query = st.text_input("Enter your query:")

with st.expander("⚙️ Advanced Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Top-K (retrieval)", min_value=1, max_value=30, value=10, step=1)
    with col2:
        top_p = st.slider("Top-P threshold (relevance)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    with col3:
        num_display = st.slider("Display N results", min_value=1, max_value=10, value=3, step=1)

# ==== Search & rerank ====
if query:
    with st.spinner("Searching..."):
        results = vectorstore.similarity_search(query, k=top_k)
    pairs = [(query, doc.page_content) for doc in results]
    with st.spinner("Reranking..."):
        raw_scores = reranker.predict(pairs)

    def sigmoid(x):
        return 1 / (1 + torch.exp(-torch.tensor(x)))
    normalized_scores = [sigmoid(score).item() for score in raw_scores]

    scored_results = list(zip(normalized_scores, results))
    scored_results.sort(reverse=True, key=lambda x: x[0])

    # Dedublicate
    unique_results = {}
    for score, doc in scored_results:
        if score < top_p:
            continue

        source = doc.metadata.get('source')
        if source not in unique_results:
            unique_results[source] = (score, doc)

        if len(unique_results) >= num_display:
            break

    if unique_results:
        st.success(f"Found {len(unique_results)} unique articles with score ≥ {top_p:.2f}:")
        st.markdown("\n---\n")
        st.subheader("🔮 GPT-4o Answer")

        gpt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and knowledgeable assistant. You should prioritize using the provided context from the retrieved articles and associated images to answer the user’s question. "
                    "You may use your general knowledge to clarify or elaborate, but avoid introducing facts that clearly contradict or go beyond the provided context. "
                    "If the provided context does not contain enough relevant information to confidently answer, say: 'Based on the provided documents, I do not have enough information to answer this question fully.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please answer the following question by primarily using the context provided below. "
                    f"You may add clarification if helpful, but do not invent information beyond what the context reasonably supports. "
                    f"If the context is insufficient, say: 'Based on the provided documents, I do not have enough information to answer this question fully.'\n\n"
                    f"Question: {query}\n\n"
                    f"Context:\n"
                )
            }
        ]

        for i, (score, doc) in enumerate(unique_results.values()):
            article_text = f"Article {i+1}: {doc.metadata['title']}\n{doc.page_content[:1000]}..."
            gpt_messages.append({"role": "user", "content": article_text})

            if doc.metadata.get('feature_image'):
                gpt_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here is an associated image for Article {i+1}."},
                        {
                            "type": "image_url",
                            "image_url": {"url": doc.metadata['feature_image']}
                        }
                    ]
                })

        if st.button("👁️ Generate Answer with GPT-4o"):
            with st.spinner("Generating answer with GPT-4o..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=gpt_messages,
                    max_tokens=1000,
                    temperature=0.0
                )
                answer = response['choices'][0]['message']['content']
                st.markdown("### 🔗 GPT-4o Answer:")
                st.write(answer)

        # Retrieved articles
        st.markdown("\n---\n")
        st.subheader("📚 Retrieved Articles")

        for i, (score, doc) in enumerate(unique_results.values()):
            st.markdown(f"### {i+1}. [{doc.metadata['title']}]({doc.metadata['source']})")
            st.markdown(f"**Relevance score (normalized):** {score:.4f}")
            st.progress(score)

            cols = st.columns([1, 2])

            with cols[0]:
                if doc.metadata.get('feature_image'):
                    st.image(doc.metadata['feature_image'], use_container_width=True)

            with cols[1]:
                st.write(doc.page_content[:1000] + "...")
    else:
        st.warning("⚠️ No relevant documents were found above the threshold. Please try a different query or adjust the advanced settings (Top-K or Top-P).")
