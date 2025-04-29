# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time 
import config

print("Starting model loading...")
start_load_time = time.time()

base_model_name = config.BASE_LLM_NAME
adapter_name = config.ADAPTER_NAME

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token.")
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added default [PAD] token as pad_token.")
print("Tokenizer loaded.")

# 2. Load Base Model 
print(f"Loading Base Model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Base Model loaded.")

# 3. Load PEFT Adapter 
print(f"Loading PEFT Adapter: {adapter_name}")
model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval() 
print("PEFT Adapter loaded and merged.")

end_load_time = time.time()
print(f"--- Model loading finished in {end_load_time - start_load_time:.2f} seconds ---")

# --- Generation Function (Call Multiple Times) ---
def generate_response(instruction_text, loaded_model, loaded_tokenizer):

    start_gen_time = time.time()

    prompt = f"### Instruction:\n{instruction_text}\n\n### Response:\n"
    inputs = loaded_tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(loaded_model.device)
    prompt_token_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = loaded_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # Include attention mask
            max_new_tokens=150,
            temperature=0.7, # You might need to adjust temp/top_p for instruction models
            top_p=0.9,
            do_sample=True,
            pad_token_id=loaded_tokenizer.pad_token_id # Use the correctly set pad_token_id
        )

    response_tokens = outputs[0][prompt_token_length:]
    response = loaded_tokenizer.decode(response_tokens, skip_special_tokens=True)

    end_gen_time = time.time()
    print(f"Generation finished in {end_gen_time - start_gen_time:.2f} seconds.")
    return response.strip()


prompt1 = "Explain the concept of recursion in computer science."
response1 = generate_response(prompt1, model, tokenizer)
print("\nResponse 1:")
print(response1)

prompt2 = "What is the difference between a list and a tuple in Python?"
response2 = generate_response(prompt2, model, tokenizer)
print("\nResponse 2:")
print(response2)

prompt3 = "Write a short Python function to calculate factorial."
response3 = generate_response(prompt3, model, tokenizer)
print("\nResponse 3:")
print(response3)