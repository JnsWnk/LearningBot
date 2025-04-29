import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_LLM_NAME = os.getenv("BASE_LLM_NAME")
ADAPTER_NAME = os.getenv("ADAPTER_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")