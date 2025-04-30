import json
import time
import torch
from pymongo.collection import Collection # For type hinting

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import schemas # If needed here, or pass data directly
import config

def interpret_intent(user_input: str, llm, tokenizer) -> dict:
    print(f"  CHATBOT: Interpreting intent for: '{user_input}'")
    # Define intents and prompt
    intent_list = "[ASK_QUESTION, REQUEST_QUIZ, SUMMARIZE_TOPIC, OTHER]"
    classifier_prompt = f"""Analyze user intent and topic. Output ONLY JSON.
Intent list: "{intent_list}"
User: "what is python?"
JSON: {{"intent": "ASK_QUESTION", "topic": "python"}}

User: "quiz me on data structures"
JSON: {{"intent": "REQUEST_QUIZ", "topic": "data structures"}}

User: "thanks"
JSON: {{"intent": "OTHER", "topic": null}}

User: "{user_input}"
JSON:"""
    try:
        inputs = tokenizer(classifier_prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_part = result_text[len(classifier_prompt):].strip()
        print(f"  CHATBOT NLU Raw Output: {json_part}")
        try:
            parsed = json.loads(json_part)
            intent = parsed.get("intent", "OTHER")
            topic = parsed.get("topic")
            # Basic validation
            if intent not in intent_list.replace("[","").replace("]","").split(", "):
                 intent="OTHER"
            print(f"  CHATBOT NLU Parsed: Intent={intent}, Topic={topic}")
            return {"intent": intent, "topic": topic}
        except json.JSONDecodeError:
            print("  CHATBOT NLU Error: Could not parse JSON response.")
            return {"intent": "OTHER", "topic": None}
    except Exception as e:
        print(f"  CHATBOT NLU Error during generation: {e}")
        return {"intent": "ERROR", "topic": None}

# --- RAG Function ---
def retrieve_context(query_text: str, embedding_model, index_name: str, k: int = 3) -> str:
    print(f"  CHATBOT: Retrieving RAG context for: '{query_text}'")
    if not query_text: return "No query provided for context retrieval."
    try:
        query_vector = embedding_model.encode(query_text).tolist()
        pipeline = [
            {'$vectorSearch': {
                'index': index_name, 'path': 'embedding_vector',
                'queryVector': query_vector,
                'numCandidates': 100, 'limit': k
            }},
            {'$project': {'_id': 0, 'text_chunk': 1, 'score': {'$meta': 'vectorSearchScore'}}}
        ]
        results = list(db.users_collection.aggregate(pipeline))
        context = "\n---\n".join([f"Context: {res['text_chunk']} (Score: {res['score']:.4f})" for res in results])
        print(f"  CHATBOT: Retrieved {len(results)} context chunks.")
        return context if context else "No relevant context found in the database."
    except Exception as e:
        print(f"  CHATBOT Error during RAG retrieval: {e}")
        return "Error retrieving context."

def format_rag_prompt(query_text: str, context: str) -> str:
    """Formats the prompt for the LLM."""
    prompt = f"""### Instruction:
Provide a comprehensive and detailed answer to the following question based on the provided context preferably and your own knowledge if applicable.
If the information from the context is not fitting or not enough and you are not sure about the answer, say so.

Context:
{context}

Question: {query_text}

### Response:
"""
    return prompt

# --- LLM Generation ---
def generate_llm_answer(prompt: str, llm, tokenizer) -> str:
    print(f"  CHATBOT: Generating LLM answer...")
    try:
        # Consider adjusting max_length based on model limits and context size
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1800).to(llm.device)
        prompt_token_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = llm.generate(
                **inputs, max_new_tokens=500, temperature=0.7, top_p=0.9,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        response_tokens = outputs[0][prompt_token_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print(f"  CHATBOT: Answer generated.")
        return response.strip()
    except Exception as e:
        print(f"  CHATBOT Error during LLM generation: {e}")
        return "Error generating response."

def process_chat_message(
    user_input: str,
    user_id: str,
    llm,
    tokenizer,
    embedding_model,
    vector_index_name: str
) -> str:
    start_time = time.time()

    intent_data = interpret_intent(user_input, llm, tokenizer)
    intent = intent_data.get("intent", "OTHER")
    topic = intent_data.get("topic")
    standardized_topic = topic.lower().replace(" ", "_") if topic else None

    bot_answer = ""
    action_taken = False
    if intent == "ASK_QUESTION":
        print("  CHATBOT: Handling ASK_QUESTION intent.")
        query = topic or user_input
        print("Query: " + query)
        context = retrieve_context(query, embedding_model, vector_index_name)
        prompt = format_rag_prompt(query, context)
        bot_answer = generate_llm_answer(prompt, llm, tokenizer)
        action_taken = True
    elif intent == "REQUEST_QUIZ":
        # Placeholder for quiz logic
        print("  CHATBOT: Handling REQUEST_QUIZ intent.")
        if topic:
            bot_answer = f"Okay, let's start a quiz on {topic}! (Quiz logic not implemented yet)."
            action_taken = True
            standardized_topic = None # Don't update profile just for starting
        else:
            bot_answer = "Which topic would you like to be quizzed on?"
            action_taken = True
    # Add elif for SUMMARIZE_TOPIC, etc.
    else: # OTHER / Fallback
        print("  CHATBOT: Handling OTHER/Fallback intent.")
        context = retrieve_context(user_input, embedding_model, vector_index_name)
        prompt = format_rag_prompt(user_input, context)
        bot_answer = generate_llm_answer(prompt, llm, tokenizer)
        action_taken = True
        standardized_topic = None # Don't assume topic learned

    if action_taken and standardized_topic and intent == "ASK_QUESTION":
        print(f"  CHATBOT: Updating knowledge profile for user {user_id}, topic '{standardized_topic}'")
        try:
            # Ensure crud.update_user_knowledge is accessible
            db.update_user_knowledge(user_id, standardized_topic, mastery_level="Seen")
        except Exception as e:
            print(f"  CHATBOT Error updating knowledge profile: {e}") # Log error

    end_time = time.time()
    print(f"Chat request processed in {end_time - start_time:.2f} seconds.")
    return bot_answer