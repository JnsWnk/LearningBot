from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import time

import config

#NLU_MODEL_ID = "google/flan-t5-base"
NLU_MODEL_ID = "JnsDev/flan-t5-chatbot-intent"


def load_all_models(emb=True, tiny=True, flan=True):
    """Loads embedding model, fine-tuned TinyLlama, and Mistral 7B Instruct."""
    models = {}
    print("CORE: Loading ML models...")
    start_time = time.time()
    load_successful = True

    # --- 1. Embedding Model ---
    if(emb):
        try:
            print(f"CORE: Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            models["emb"] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
            print(f"CORE: Embedding model loaded on {device}.")
        except Exception as e:
            print(f"CORE ERROR loading embedding model: {e}")
            models["embedding_model_error"] = str(e)
            load_successful = False

    # --- 2. Fine-tuned TinyLlama (llm_mini) ---
    if(tiny):
        try:
            print(f"CORE: Loading TinyLlama: {config.BASE_LLM_NAME} + adapter {config.ADAPTER_NAME}...")
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            tokenizer_mini = AutoTokenizer.from_pretrained(config.BASE_LLM_NAME)
            if tokenizer_mini.pad_token is None:
                tokenizer_mini.pad_token = tokenizer_mini.eos_token
            models["tok_mini"] = tokenizer_mini

            quantization_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            print(f"CORE: Loading TinyLlama base with 4-bit quantization (Compute: {compute_dtype})...")
            base_model_mini = AutoModelForCausalLM.from_pretrained(
                config.BASE_LLM_NAME,
                quantization_config=quantization_config_4bit,
                device_map="auto",
            )
            print("CORE: TinyLlama base model loaded.")

            print(f"CORE: Loading PEFT adapter: {config.ADAPTER_NAME}...")
            llm_mini = PeftModel.from_pretrained(base_model_mini, config.ADAPTER_NAME)
            llm_mini.eval()
            models["mini"] = llm_mini
            print("CORE: TinyLlama PEFT Adapter loaded.")

        except Exception as e:
            print(f"CORE ERROR loading TinyLlama model or adapter: {e}")
            models["mini_error"] = str(e)
            load_successful = False

    if(flan):
        try:
            print(f"CORE: Loading NLU model: {NLU_MODEL_ID}...")
            # Load on CPU by default to save VRAM, can change to 'cuda' if preferred/available
            nlu_device = 'cuda' if torch.cuda.is_available() else 'cpu' # Or force 'cpu'
            print(f"CORE: Attempting to load NLU model from: {NLU_MODEL_ID}") # Add this
            nlu_tokenizer = AutoTokenizer.from_pretrained(NLU_MODEL_ID)
            nlu_model = AutoModelForSeq2SeqLM.from_pretrained(NLU_MODEL_ID).to(nlu_device)
            print(f"CORE: Successfully loaded NLU model class: {type(nlu_model)}") # Add this
        
            nlu_model.eval() # Set to evaluation mode

            models["nlu"] = nlu_model
            models["tok_nlu"] = nlu_tokenizer
            print(f"CORE: NLU model loaded on {nlu_device}.")

        except Exception as e:
            print(f"CORE CRITICAL ERROR loading NLU model: {e}")
            models["nlu_error"] = str(e)
            load_successful = False

    # --- Finalize ---
    end_time = time.time()
    # Adjust success check if Mistral is now essential
    if load_successful:
        print(f"CORE: Finished loading all required ML models in {end_time - start_time:.2f} seconds.")
    else:
         print(f"CORE: WARNING - Some models failed to load. Check errors above. Time taken: {end_time - start_time:.2f} seconds.")
    return models