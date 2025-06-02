from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import pickle
import json
import argparse
from tqdm import tqdm


CHUNK_SIZE = 1000 
CHUNK_OVERLAP=200


class ArticleVectorStoreBuilder:
    def __init__(self, name: str, articles: list, embeddings, base_path: str = "vectorstores"):
        self.name = name
        self.articles = articles
        self.embeddings = embeddings
        self.base_path = base_path
        self.full_path = os.path.join(self.base_path, self.name)

        os.makedirs(self.full_path, exist_ok=True)
        self.tracked_ids = set()
        self.vectorstore = None

    def load_existing_index(self):
        faiss_path = os.path.join(self.full_path, "faiss_index")
        tracked_ids_path = os.path.join(self.full_path, "tracked_ids.pkl")

        if os.path.exists(faiss_path):
            self.vectorstore = FAISS.load_local(faiss_path, embeddings=self.embeddings)
            if os.path.exists(tracked_ids_path):
                with open(tracked_ids_path, 'rb') as f:
                    self.tracked_ids = pickle.load(f)

    def compute_missing_articles(self):
        missing = [article for article in self.articles if article['metadata']['url'] not in self.tracked_ids]
        print(f"Found {len(missing)} new articles to index.")
        return missing

    def process_articles(self, missing_articles):
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        missing_docs = []

        for article in tqdm(missing_articles, desc="Indexing articles"):
            text = f"{article['metadata']['title']}\n\n{article['text']}"
            doc = Document(page_content=text, metadata={
                "source": article['metadata']['url'],
                "title": article['metadata']['title'],
                "feature_image": article['metadata']['feature_image']
            })
            chunks = text_splitter.split_documents([doc])
            missing_docs.extend(chunks)

        print(f"Adding {len(missing_docs)} chunks to the vectorstore...")
        return missing_docs

    def save_index(self):
        faiss_path = os.path.join(self.full_path, "faiss_index")
        tracked_ids_path = os.path.join(self.full_path, "tracked_ids.pkl")

        self.vectorstore.save_local(faiss_path)

        with open(tracked_ids_path, 'wb') as f:
            pickle.dump(self.tracked_ids, f)

    def run(self):
        self.load_existing_index()
        missing_articles = self.compute_missing_articles()
        missing_docs = self.process_articles(missing_articles)
        new_vectorstore = FAISS.from_documents(missing_docs, embedding=self.embeddings)

        if self.vectorstore is None:
            self.vectorstore = new_vectorstore
        else:
            self.vectorstore.merge_from(new_vectorstore)

        self.tracked_ids.update(article['metadata']['url'] for article in missing_articles)
        self.save_index()
        return self.vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vectorstore from articles.")
    parser.add_argument("--name", type=str, default="my_articles_store", help="Name of the vectorstore")
    parser.add_argument("--articles", type=str, default="all_articles.json", help="Path to articles JSON file")
    args = parser.parse_args()

    print(f"Using vectorstore name: {args.name}")
    print(f"Loading articles from: {args.articles}")

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    with open(args.articles, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    builder = ArticleVectorStoreBuilder(
        name=args.name,
        articles=articles,
        embeddings=embeddings
    )
    vectorstore = builder.run()
    print("Vectorstore build completed!")