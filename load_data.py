import os
import time
import torch 
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm 
import backend.config as config

MONGO_CONNECTION_STRING = config.MONGO_CONNECTION_STRING
DB_NAME = config.DB_NAME
COLLECTION_NAME = config.DOCS_COLLECTION 

EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME

# Dataset for RAG
DATASET_WIKI_AI_ML = "lmassaron/Wikipedia-AI-ML"
DATASET_WIKI_CS = "AlaaElhilo/Wikipedia_ComputerScience"
DATASET_TEXTBOOKS = "open-phi/textbooks"

# Filter for datasets
TEXTBOOKS_FILTER_FIELD = "field" 
TEXTBOOKS_RELEVANT_TOPICS = [
    "Computer Science", "Programming", "Data Science",
    "Artificial Intelligence", "Machine Learning", "Algorithms",
    "Software Engineering", "Web Development", "Database", "Operating Systems"
]

TEXTBOOKS_RELEVANT_TOPICS_LC = [topic.lower() for topic in TEXTBOOKS_RELEVANT_TOPICS]

# Text Chunking Parameters
CHUNK_SIZE = 800 # Characters per chunk
CHUNK_OVERLAP = 100 # Characters overlap between chunks

LIMIT_WIKI_AI_ML = 10000 # Max articles from Wikipedia-AI-ML
LIMIT_WIKI_CS = 10000    # Max articles from Wikipedia_ComputerScience
LIMIT_TEXTBOOKS = 500    # Max *filtered* textbook documents
MAX_TOTAL_CHUNKS = 75000 # Overall safety limit for chunks

# MongoDB Batch Insert Size
BATCH_SIZE = 200

# MongoDB Connection
try:
    print("Connecting to MongoDB Atlas...")
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    client.admin.command('ping')
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f"Embedding model loaded on {device}.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

def load_wiki_ai_ml(limit):
    print(f"\nLoading dataset: {DATASET_WIKI_AI_ML} (limit: {limit})")
    try:
        dataset = load_dataset(DATASET_WIKI_AI_ML, split=f'train[:{limit}]') # Load only up to the limit
        docs = []
        for i, item in enumerate(dataset):
            text = item.get("text")
            if text:
                metadata = {"source": DATASET_WIKI_AI_ML, "doc_id": f"wiki_ai_ml_{i}"}
                docs.append({"text": text, "metadata": metadata})
        print(f"Loaded {len(docs)} documents from {DATASET_WIKI_AI_ML}")
        return docs
    except Exception as e:
        print(f"Error loading {DATASET_WIKI_AI_ML}: {e}")
        return []

def load_wiki_cs(limit):
    print(f"\nLoading dataset: {DATASET_WIKI_CS} (limit: {limit})")
    try:
        dataset = load_dataset(DATASET_WIKI_CS, split=f'train[:{limit}]')
        docs = []
        for i, item in enumerate(dataset):
            text = item.get("text")
            if text:
                 metadata = {"source": DATASET_WIKI_CS, "doc_id": f"wiki_cs_{i}"}
                 docs.append({"text": text, "metadata": metadata})
        print(f"Loaded {len(docs)} documents from {DATASET_WIKI_CS}")
        return docs
    except Exception as e:
        print(f"Error loading {DATASET_WIKI_CS}: {e}")
        return []

def load_filtered_textbooks(limit, topics_lc):
    print(f"\nLoading and filtering dataset: {DATASET_TEXTBOOKS} (limit: {limit})")
    try:
        # Use streaming to avoid downloading the whole dataset
        dataset_stream = load_dataset(DATASET_TEXTBOOKS, split='train', streaming=True)
        docs = []
        processed_count = 0
        print(f"Filtering for topics in field 'field': {TEXTBOOKS_RELEVANT_TOPICS}")

        for item in dataset_stream:
             if processed_count >= limit:
                 break

             field_value = item.get(TEXTBOOKS_FILTER_FIELD)
             if field_value:
                 if any(topic in str(field_value).lower() for topic in topics_lc):
                     content = item.get("markdown") 
                     if content:
                         metadata = {"source": DATASET_TEXTBOOKS}
                         metadata["topic"] = item.get("topic", "Unknown")
                         metadata["field"] = field_value
                         metadata["doc_id"] = f"textbook_{processed_count}"

                         docs.append({"text": content, "metadata": metadata})
                         processed_count += 1

        print(f"Loaded and filtered {len(docs)} documents from {DATASET_TEXTBOOKS}")
        return docs
    except Exception as e:
        print(f"Error loading or filtering {DATASET_TEXTBOOKS}: {e}")
        print(f"Please verify the dataset structure, especially the filter field 'field' and content field ('markdown').")
        return []

start_time_total = time.time()

raw_docs_ai_ml = load_wiki_ai_ml(LIMIT_WIKI_AI_ML)
raw_docs_cs = load_wiki_cs(LIMIT_WIKI_CS)
raw_docs_textbooks = load_filtered_textbooks(LIMIT_TEXTBOOKS, TEXTBOOKS_RELEVANT_TOPICS_LC)

all_raw_docs = raw_docs_ai_ml + raw_docs_cs + raw_docs_textbooks
print(f"\nTotal documents loaded across sources: {len(all_raw_docs)}")

mongo_batch = []
total_chunks_processed = 0
print("\nStarting chunking, embedding, and uploading...")

for doc in tqdm(all_raw_docs, desc="Processing documents"):
    if total_chunks_processed >= MAX_TOTAL_CHUNKS:
        print(f"\nReached overall chunk limit: {MAX_TOTAL_CHUNKS}. Stopping.")
        break

    text = doc["text"]
    metadata = doc["metadata"]

    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        if total_chunks_processed >= MAX_TOTAL_CHUNKS:
            break 

        try:
            embedding = embedding_model.encode(chunk).tolist()

            mongo_doc = {
                "text_chunk": chunk,
                "embedding_vector": embedding,
                "metadata": metadata.copy() 
            }
            mongo_doc["metadata"]["chunk_index"] = i

            mongo_batch.append(mongo_doc)
            total_chunks_processed += 1

            if len(mongo_batch) >= BATCH_SIZE:
                try:
                    collection.insert_many(mongo_batch)
                except Exception as e:
                    tqdm.write(f"Error inserting batch: {e}") # Use tqdm.write inside loop
                mongo_batch = [] # Clear batch

        except Exception as e:
            tqdm.write(f"Error embedding or preparing chunk {i} for doc {metadata.get('doc_id', 'N/A')}: {e}")


# Insert any remaining documents in the last batch
if mongo_batch:
    print(f"\nInserting final batch of {len(mongo_batch)} chunks...")
    try:
        collection.insert_many(mongo_batch)
    except Exception as e:
        print(f"Error inserting final batch: {e}")

end_time_total = time.time()
print("\n--- Processing Complete ---")
print(f"Total documents processed from sources: {len(all_raw_docs)}")
print(f"Total chunks generated and attempted upload: {total_chunks_processed}")
print(f"Total time: {end_time_total - start_time_total:.2f} seconds")
print("\nIMPORTANT: Data loading complete.")
print("Please go to your MongoDB Atlas UI and create a Vector Search Index on the")
print(f"'{DB_NAME}.{COLLECTION_NAME}' collection, targeting the 'embedding_vector' field.")
print("Use dimensions=384 and similarity=cosine (or your preferred metric).")

client.close()