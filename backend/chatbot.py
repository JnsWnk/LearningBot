import json
import time
import torch
import requests
from pymongo.collection import Collection # For type hinting

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import prompts
import schemas # If needed here, or pass data directly
import config

# API Configuration
API_CONFIG = {
    "base_url": "https://genai.hkbu.edu.hk/general/rest",
    "model": "gpt-4-o",
    "api_version": "2024-10-21"
}


def interpret_intent(user_input: str, nlu_model, nlu_tokenizer, use_gpt4: bool = False) -> dict:
    if use_gpt4:
        intent_list = ["GET_INFORMATION", "MANAGE_KNOWLEDGE", "REQUEST_REVIEW", "OTHER"]
        messages = prompts.create_nlu_prompt_messages(user_input, intent_list)
        result = generate_response(messages, nlu_model, nlu_tokenizer, True)
        try: 
            parsed = json.loads(result)
            intent = parsed.get("intent", "OTHER")
            topic = parsed.get("topic")
            if intent not in intent_list:
                intent = "OTHER"
            return {"intent": intent, "topic": topic}
        except json.JSONDecodeError:
            return {"intent": "OTHER", "topic": None}
    else:
        return interpret_intent_flan(user_input, nlu_model, nlu_tokenizer)

def interpret_intent_flan(user_input: str, nlu_model, nlu_tokenizer) -> dict:
    PREFIX = "Classify intent and extract topic: "
    prompt = f"{PREFIX}{user_input}"
    print(f"  NLU Prompt: {prompt}")

    try:
        inputs = nlu_tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(nlu_model.device)

        with torch.no_grad():
            outputs = nlu_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1, 
                do_sample=False
            )
        result_text = nlu_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"  CHATBOT NLU Raw Output (FLAN-T5): '{result_text}'")

        if ':' in result_text and '"' in result_text:
            json_string_to_parse = "{" + result_text + "}"
            print(f"  CHATBOT NLU Attempting to parse: '{json_string_to_parse}'")
            try:
                parsed = json.loads(json_string_to_parse)
                intent = parsed.get("intent", "OTHER")
                topic = parsed.get("topic")

                valid_intents = ["GET_INFORMATION", "MANAGE_KNOWLEDGE", "REQUEST_REVIEW", "OTHER"]
                if intent not in valid_intents:
                    print(f"  CHATBOT NLU Warning: Parsed intent '{intent}' not in valid list. Defaulting to OTHER.")
                    intent = "OTHER"

                print(f"  CHATBOT NLU Parsed: Intent={intent}, Topic={topic}")
                return {"intent": intent, "topic": topic}

            except json.JSONDecodeError as json_err:
                print(f"  CHATBOT NLU Error: JSONDecodeError after adding braces: {json_err}")
                print(f"  String that failed parsing: '{json_string_to_parse}'")
                return {"intent": "OTHER", "topic": None}
        else:
            print("  CHATBOT NLU Warning: Output doesn't look like key-value pairs. Defaulting to OTHER.")
            return {"intent": "OTHER", "topic": None}

    except Exception as e:
        print(f"  CHATBOT NLU Error during generation or processing: {e}")
        return {"intent": "ERROR", "topic": None}

def retrieve_context(query_text: str, embedding_model, index_name: str, k: int = 3) -> str:
    if not query_text:
        return "No query provided for context retrieval."
    try:
        query_vector = embedding_model.encode(query_text).tolist()
        pipeline = [
            {'$vectorSearch': {
                'index': index_name,
                'path': 'embedding_vector',
                'queryVector': query_vector,
                'numCandidates': 100,
                'limit': k
            }},
            {'$project': {'_id': 0, 'text_chunk': 1, 'score': {'$meta': 'vectorSearchScore'}}}
        ]
        results = list(db.docs_collection.aggregate(pipeline))
        context = "\n---\n".join([f"Context: {res['text_chunk']} (Score: {res['score']:.4f})" for res in results])
        print("Context found in RAG DB: ", len(results))
        return context if context else "No relevant context found in the database."
    except Exception as e:
        print(f"Error during RAG retrieval: {e}")
        return "Error retrieving context."

def generate_response(prompt, model, tokenizer, use_gpt4=True) -> str:
    try:
        if use_gpt4:
            try:
                url = f"{API_CONFIG['base_url']}/deployments/{API_CONFIG['model']}/chat/completions/?api-version={API_CONFIG['api_version']}"
                headers = {'Content-Type': 'application/json', 'api-key': config.GPT_KEY}
                payload = {'messages': prompt}
                
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Error during GPT-4 generation: {e}")
                return "Error generating response from GPT-4."
        
        else:
            print("  GENERATOR: Using local TinyLlama model")
            if model is None or tokenizer is None:
                 print("Error: Local model or tokenizer not provided for generation.")
                 return "Error: Local model not available."

            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1800).to(model.device)
                prompt_token_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500,      
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id 
                    )

                response_tokens = outputs[0][prompt_token_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                return response.strip()

            except Exception as e:
                import traceback
                print(f"Error during local LLM generation: {e}")
                traceback.print_exc() 
                return "Error generating response from local model."

    except Exception as e:
        import traceback
        print(f"Unexpected error in generate_response function: {e}")
        traceback.print_exc()
        return "An unexpected error occurred during generation."

def process_chat_message(
    user_input: str,
    user_id: str,
    models,
    vector_index_name,
    use_gpt4: bool = True
) -> str:
    start_time = time.time()
    
    chat_history = db.get_chat_history(user_id, limit=5)
    
    user_data = db.get_user_by_id(user_id)
    if not user_data:
        print(f"Warning: Could not find user data for ID {user_id}")
        user_data = {"knowledge_profile": {}, "learning_goals": []}
    else:
        print(f"Found user data for ID {user_id}: {user_data.get('knowledge_profile', {})}")
    
    intent_data = interpret_intent(user_input, models["nlu"], models["tok_nlu"], use_gpt4=False)
    intent = intent_data.get("intent", "OTHER")
    topic = intent_data.get("topic")
    standardized_topic = topic.lower().replace(" ", "_") if topic else None
    use_gpt4 = True
    query = topic or user_input
    context = retrieve_context(query, models["emb"], vector_index_name)

    if intent == "GET_INFORMATION":
        if use_gpt4:
            prompt = prompts.create_get_information_prompt(user_input, context, chat_history)
        else:
            prompt = prompts.create_get_information_prompt_tinyl(user_input, context, chat_history)
        bot_answer = generate_response(prompt, models["mini"], models["tok_mini"], use_gpt4)
        
        if standardized_topic:
            try:
                db.update_user_knowledge(
                    user_id=user_id,
                    concept=standardized_topic,
                    mastery_level="Seen"
                )
            except Exception as e:
                print(f"Error updating knowledge profile: {e}")
                
    elif intent == "MANAGE_KNOWLEDGE":
        if not topic:
            bot_answer = "I need to know which topic you'd like to manage. Please specify a topic."
        else:
            prompt = prompts.create_manage_knowledge_prompt(topic, context, user_data, chat_history)
            bot_answer = generate_response(prompt, models["mini"], models["tok_mini"], True)
            
            try:
                db.update_user_knowledge(
                    user_id=user_id,
                    concept=standardized_topic,
                    mastery_level="Practiced"
                )
            except Exception as e:
                print(f"Error updating knowledge profile: {e}")
                
    elif intent == "REQUEST_REVIEW":
        if not topic:
            print("User data for review:", user_data.get('knowledge_profile', {}))
            knowledge_profile = user_data.get("knowledge_profile", {})
            if knowledge_profile:
                topic_to_review = min(
                    knowledge_profile.items(),
                    key=lambda x: {
                        "Seen": 0,
                        "Practiced": 1,
                        "Assessed-Low": 2,
                        "Assessed-High": 3,
                        "Proficient": 4
                    }.get(x[1].get("mastery", "Seen"), 0)
                )[0]
                topic = topic_to_review.replace("_", " ")
                standardized_topic = topic.lower().replace(" ", "_") if topic else None
            else:
                bot_answer = "I don't have any topics to review yet. Try learning about a topic first!"
                return bot_answer
        
        review_context = retrieve_context(topic, models["emb"], vector_index_name)
        prompt = prompts.create_request_review_prompt(topic, review_context, user_data, chat_history)
        bot_answer = generate_response(prompt, models["mini"], models["tok_mini"], True)
        
        try:
            db.update_user_knowledge(
                user_id=user_id,
                concept=standardized_topic,
                mastery_level="Assessed-Low"
            )
        except Exception as e:
            print(f"Error updating knowledge profile: {e}")
            
    else:
        prompt = prompts.create_other_prompt(user_input, chat_history, user_data)
        bot_answer = generate_response(prompt, models["mini"], models["tok_mini"], True)

    try:
        db.save_chat_history(user_id, user_input, bot_answer)
    except Exception as e:
        print(f"Error saving chat history: {e}")

    print(f"Chat request processed in {time.time() - start_time:.2f} seconds.")
    return bot_answer