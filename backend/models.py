from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import time

import config

MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_all_models():
    """Loads embedding model, fine-tuned TinyLlama, and Mistral 7B Instruct."""
    models = {}
    print("CORE: Loading ML models...")
    start_time = time.time()
    load_successful = True

    # --- 1. Embedding Model ---
    try:
        print(f"CORE: Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models["embedding_model"] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        print(f"CORE: Embedding model loaded on {device}.")
    except Exception as e:
        print(f"CORE ERROR loading embedding model: {e}")
        models["embedding_model_error"] = str(e)
        load_successful = False

    # --- 2. Fine-tuned TinyLlama (llm_mini) ---
    try:
        print(f"CORE: Loading TinyLlama: {config.BASE_LLM_NAME} + adapter {config.ADAPTER_NAME}...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        tokenizer_mini = AutoTokenizer.from_pretrained(config.BASE_LLM_NAME)
        if tokenizer_mini.pad_token is None:
            tokenizer_mini.pad_token = tokenizer_mini.eos_token
        models["tokenizer_mini"] = tokenizer_mini

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
        models["llm_mini"] = llm_mini
        print("CORE: TinyLlama PEFT Adapter loaded.")

    except Exception as e:
        print(f"CORE ERROR loading TinyLlama model or adapter: {e}")
        models["llm_mini_error"] = str(e)
        load_successful = False


    # --- 3. Mistral 7B Instruct (mistral_model) ---
    if torch.cuda.is_available(): # Quantization primarily benefits GPU usage
        try:
            print("-" * 20)
            print(f"CORE: Loading Mistral Instruct: {MISTRAL_MODEL_ID}...")

            # Load Mistral tokenizer
            tokenizer_mistral = AutoTokenizer.from_pretrained(MISTRAL_MODEL_ID)
            # Mistral tokenizer might handle padding differently or implicitly
            if tokenizer_mistral.pad_token is None:
                 # Common practice for Mistral, check model card if issues arise
                 tokenizer_mistral.pad_token = tokenizer_mistral.eos_token
            models["tokenizer_mistral"] = tokenizer_mistral

            # Define quantization config (can reuse if compute_dtype determined correctly above)
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            quantization_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            print(f"CORE: Loading Mistral base with 4-bit quantization (Compute: {compute_dtype})...")
            mistral_model = AutoModelForCausalLM.from_pretrained(
                MISTRAL_MODEL_ID,
                quantization_config=quantization_config_4bit,
                device_map="auto",
                # token=... # Add token if needed for specific versions, but usually not for base Mistral
            )
            mistral_model.eval()
            models["mistral_model"] = mistral_model # Use new key
            print("CORE: Mistral Instruct model loaded.")
            print("-" * 20)

        except ImportError:
             print("CORE ERROR: bitsandbytes library not found for Mistral. Please install it: pip install bitsandbytes")
             models["mistral_error"] = "bitsandbytes not installed"
             load_successful = False
        except Exception as e:
            print(f"CORE CRITICAL ERROR loading Mistral model: {e}")
            models["mistral_error"] = str(e)
            load_successful = False
    else:
        print("CORE WARNING: CUDA not available, skipping Mistral 7B loading as quantization requires GPU.")
        models["mistral_error"] = "CUDA not available"
        # load_successful = False # Uncomment if Mistral is essential

    # --- Finalize ---
    end_time = time.time()
    # Adjust success check if Mistral is now essential
    if load_successful and "mistral_model" in models:
        print(f"CORE: Finished loading all required ML models in {end_time - start_time:.2f} seconds.")
    else:
         print(f"CORE: WARNING - Some models failed to load. Check errors above. Time taken: {end_time - start_time:.2f} seconds.")
    return models