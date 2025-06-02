# RAG-for-The-Batch
Multimodal Retrieval-Augmented  Generation (RAG) System that retrieves relevant news articles from The Batch - https://www.deeplearning.ai/the-batch/

**Important:** check Task report.pdf file. You can also see the [Demo](https://drive.google.com/file/d/1nj5u7Mmzbsg2KEhs15PwA8X92NszZDnE/view?usp=drive_link)

## 🚀 Technology Stack

- **Python 3.13**
- **FAISS** — fast similarity search
- **HuggingFace Sentence-Transformers** (`all-MiniLM-L6-v2`)
- **CrossEncoder** (`ms-marco-MiniLM-L-6-v2`)
- **GPT-4o** — for answer generation (via OpenAI API)
- **Streamlit** — interactive web UI
- **BeautifulSoup** + **requests** — for data ingestion
- **tqdm**, **pandas**, **numpy** — utilities and data handling

## Project Structure
```plaintext
RAG-for--Batch/
├── src/
│   ├── data_scraper/
│   │   ├── articles_scraper.py   # Utilities for scraping articles
│   │   ├── scrap.py              # Main scraping script
│   │
│   ├── indexer.py                # Build / update FAISS vector index
│   ├── app.py                    # Streamlit UI application
│   ├── evaluator.py              # Evaluation pipeline (Hit@K, MRR)
│
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
```

## How to setup?
### 1. Clone the repository
```
git clone https://github.com/znak314/RAG-for-The-Batch.git
cd RAG-for-The-Batch
```

### 2️. Create a virtual environment and install dependencies

```bash
python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Add articles
You have two options to prepare the articles dataset:

#### Option 1 — Automatically scrape articles (recommended if you want latest data)

You can run the provided script with customizable arguments:

```bash
python src/data_scraper/scrap.py --output all_articles.json
```

#### Option 2 — Download prepared dataset
If you prefer, you can download a pre-scraped dataset and place it in the project root:

👉 [Download all_articles.json from Google Drive](https://drive.google.com/file/d/1z-iGglYMNJc8Nv8pYC6x5Q4L5enOmkt6/view?usp=drive_link)

### 4️. Build vector index

Once you have prepared the articles (`all_articles.json`), you need to build the FAISS vector index.  
You have two options:

---

#### Option 1 — Build index from articles (recommended)

Run the provided indexer script:

```bash
python src/indexer.py
```

#### Option 2 — Download prepared vectorstore
If you want to skip building the index, you can download a ready-to-use vectorstore:

👉 [Download vectorstore from Google Drive](https://drive.google.com/drive/folders/1t5JhUMtWLUxTqoFB7d-0lp-hbgMo_Ofw?usp=drive_link)

Extract the contents into the /vectorstore/ folder.

### 5. Create .env and add ChatGPT API key
In order to use **GPT-4o** for answer generation, you need to provide your OpenAI API key.
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

After all, you project should look like: 

```plaintext
RAG-for--Batch/
├── src/
│   ├── data_scraper/
│   │   ├── articles_scraper.py   # Utilities for scraping articles
│   │   ├── scrap.py              # Main scraping script
│   │
│   ├── indexer.py                # Build / update FAISS vector index
│   ├── app.py                    # Streamlit UI application
│   ├── evaluator.py              # Evaluation pipeline (Hit@K, MRR)
│
├── all_articles.json             # Full scraped articles dataset (JSON)
├── vectorstores/                 # FAISS index + metadata
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── .env                          # Environmental variables
```

### 6. Run streamlit project
After completing the previous steps, you can now launch the Streamlit application:

```bash
streamlit run src/app.py
```

### 7. (Optional) Run evaluation script
Download test data from [Google drive](https://drive.google.com/file/d/1tZGZvZk3ZL7rJ7mrQwBOhqh7kyHNtTtQ/view?usp=drive_link)

Run the evaluator:

```bash
python src/evaluator.py
```
