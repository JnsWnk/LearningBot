# 02_chatbot_app_mongo.py

import time
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config


VECTOR_INDEX_NAME = config.VECTOR_INDEX_NAME

NUM_DOCS_TO_RETRIEVE = 3 # How many relevant chunks to fetch

# --- Initialization Functions ---

def connect_mongo():
    print("Connecting to MongoDB Atlas...")
    try:
        client = MongoClient(config.MONGO_CONNECTION_STRING)
        db = client[config.DB_NAME]
        collection = db[config.COLLECTION_NAME]
        # Test connection
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        exit()

def load_embedding_model():
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        print(f"Embedding model loaded on {device}.")
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        exit()

def load_llm():
    """Loads the fine-tuned PEFT LLM and tokenizer."""
    print("Loading fine-tuned LLM...")
    start_load_time = time.time()
    try:
        # Load tokenizer (from base, consistent with indexing)
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token.")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_LLM_NAME,
            torch_dtype=torch.float16, # Use float16 for efficiency
            device_map="auto"         # Automatically map to available devices
        )
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, config.ADAPTER_NAME)
        model.eval() # Set to evaluation mode
        end_load_time = time.time()
        print(f"LLM loaded in {end_load_time - start_load_time:.2f} seconds.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading LLM: {e}")
        exit()

# --- RAG and Generation Functions ---

def get_rag_context(query_text, embedding_model, collection, index_name, k=3):
    try:
        query_vector = embedding_model.encode(query_text).tolist()

        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name,
                    'path': 'embedding_vector', # Field containing vectors
                    'queryVector': query_vector,
                    'numCandidates': 100,       # Number of candidates to consider
                    'limit': k                  # Number of top results to return
                }
            },
            { # Project only the needed fields and the search score
                '$project': {
                    '_id': 0,
                    'text_chunk': 1,
                    # 'metadata': 1, # Optionally include metadata if needed
                    'score': { '$meta': 'vectorSearchScore' }
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        for result in results:
            print("Rag result: " + str(result))
        # Format context
        context = "\n---\n".join([
            f"Context Chunk (Score: {res['score']:.4f}):\n{res['text_chunk']}"
            for res in results
        ])
        return context if context else "No relevant context found in the database."

    except Exception as e:
        print(f"Error during RAG retrieval: {e}")
        return "Error retrieving context."

def format_prompt(query_text, context):
    """Formats the prompt for the LLM using the specific instruction format."""
    # This matches the format you specified: "### Instruction:\n{instruction}\n\n### Response:\n"
    # We'll use the context to help answer the instruction (query).
    prompt = f"""### Instruction:
Answer the following question based on the provided context. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query_text}

### Response:
"""
    return prompt

def generate_answer(prompt, llm, tokenizer):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024).to(llm.device) # Truncate long prompts
        prompt_token_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=250,       # Adjust length as needed
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the newly generated tokens
        response_tokens = outputs[0][prompt_token_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- Main Chat Loop ---
if __name__ == "__main__":
    # Initialize components
    mongo_collection = connect_mongo()
    embed_model = load_embedding_model()
    llm_model, llm_tokenizer = load_llm()

    print("\n--- RAG Chatbot Initialized ---")
    print("Ask a question about CS, DS, or AI (type 'quit' to exit):")

    while True:
        user_query = input("> ")
        if user_query.lower() == 'quit':
            break
        if not user_query:
            continue

        print("Thinking...")
        start_time = time.time()

        # 1. Retrieve context
        print("  Retrieving context...")
        retrieved_context = get_rag_context(
            user_query,
            embed_model,
            mongo_collection,
            VECTOR_INDEX_NAME,
            k=NUM_DOCS_TO_RETRIEVE
        )
        # print(f"\nRetrieved Context:\n{retrieved_context}\n") # Optional: Uncomment to see context

        # 2. Format prompt
        print("  Formatting prompt...")
        final_prompt = format_prompt(user_query, retrieved_context)
        # print(f"\nFormatted Prompt:\n{final_prompt}\n") # Optional: Uncomment to see full prompt

        # 3. Generate answer
        print("  Generating answer...")
        answer = generate_answer(final_prompt, llm_model, llm_tokenizer)
        end_time = time.time()

        print("\nBot Response:")
        print(answer)
        print(f"(Time taken: {end_time - start_time:.2f} seconds)")
        print("-" * 20)

    print("Goodbye!")
    # Optional: Close MongoDB connection if needed
    # client.close()