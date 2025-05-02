from typing import List, Dict, Optional, Any
import datetime 

def format_chat_history(chat_history: List[Dict[str, Any]], max_turns: int = 2) -> List[Dict[str, str]]:
    """
    Formats chat history from database format to role-based format for prompts.
    Database format: [{"timestamp": datetime, "prompt": str, "answer": str}]
    Output format: [{"role": str, "content": str}]
    Always returns at most the last 2 user-assistant interactions (4 messages total).
    """
    formatted_history = []
    num_pairs = 0
    for i in range(len(chat_history) - 1, -1, -1):
        if num_pairs >= max_turns:
            break
            
        entry = chat_history[i]
        formatted_history.insert(0, {
            "role": "assistant",
            "content": entry["answer"]
        })
        formatted_history.insert(0, {
            "role": "user",
            "content": entry["prompt"]
        })
        num_pairs += 1

    return formatted_history


# --- Prompt Creation Functions ---

def create_get_information_prompt_tinyl(
    user_message: str,
    rag_context: str,
    chat_history: List[Dict[str, str]],
) -> str:
    formatted_history = format_chat_history(chat_history, max_turns=1)
    
    history_str = ""
    if formatted_history:
        history_str = "Chat History (Recent):\n"
        for msg in formatted_history:
            history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

    prompt = f"""### Instruction:
You are an AI tutor explaining a Computer Science, Data Science, or AI topic.
Use the 'Retrieved Context' below to answer the 'User Request'.
If the request is a follow-up, use the 'Chat History' to understand the context of the request.
Answer clearly and concisely, suitable for a student.
If the context doesn't contain the answer, state that clearly.
If the user asks to summarize, provide a concise summary based ONLY on the context.

{history_str}
Retrieved Context:
---
{rag_context}
---

User Request: {user_message}
### Response:
"""
    return prompt

def create_get_information_prompt(
    user_message: str,
    rag_context: str,
    chat_history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Creates the prompt messages for answering questions or summarizing, using RAG and history.
    Uses only the last 2 user-assistant interactions for context.
    """
    system_prompt = """You are an expert AI tutor specializing in Computer Science, Data Science, and AI.
Your goal is to provide clear, accurate, step-by-step explanations tailored for a student.
Base your answer *primarily* on the provided "Retrieved Context".
If the user asks a follow-up question (like "tell me more", "why?", "explain that part"), use the "Chat History" to understand what "that" refers to, but still ground your explanation in the "Retrieved Context" if possible.
If the context does not contain the answer, state that you cannot answer from the provided information.
If the user asks for a summary, provide a concise summary based *only* on the context."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    
    user_prompt = f"""Retrieved Context:
---
{rag_context}
---

User Request: "{user_message}"

Please provide the response based on the instructions."""
    
    messages.append({"role": "user", "content": user_prompt})
    return messages


def create_manage_knowledge_prompt(
    topic: str,
    rag_context: str,
    user_data: Dict[str, Any],
    chat_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Creates a prompt for generating a knowledge assessment question.
    Uses only the last 2 user-assistant interactions for context.
    """
    knowledge_level = "Beginner"
    if topic:
        topic_key = topic.lower().replace(" ", "_")
        if user_data.get("knowledge_profile", {}).get(topic_key):
            mastery = user_data["knowledge_profile"][topic_key].get("mastery")
            if mastery in ["Practiced", "Assessed-Low"]:
                knowledge_level = "Intermediate"
            elif mastery in ["Assessed-High", "Proficient"]:
                knowledge_level = "Advanced"

    system_prompt = f"""You are an AI quiz creator for Computer Science, Data Science, and AI topics.
Your task is to generate ONE relevant, clear question based *only* on the provided context text.
The question should be suitable for a student whose current understanding of '{topic}' is estimated as '{knowledge_level}'.
The question should test understanding, not just memorization.
Use the chat history to understand the context of the student's learning journey.
Output *only* the question text, ready to present to the student."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    
    user_prompt = f"""Context about '{topic}':
<context>
{rag_context}
</context>

Generate one question appropriate for a student at the '{knowledge_level}' level based only on the context above.
The question should help assess and improve their understanding of the topic.
Consider the student's recent learning context from the chat history."""

    messages.append({"role": "user", "content": user_prompt})
    return messages


def create_request_review_prompt(
    review_topic: str,
    rag_context: str,
    user_data: Dict[str, Any],
    chat_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Creates a prompt for generating a review question.
    Uses only the last 2 user-assistant interactions for context.
    """
    topic_key = review_topic.lower().replace(" ", "_")
    current_mastery = user_data.get("knowledge_profile", {}).get(topic_key, {}).get("mastery", "Seen")
    
    system_prompt = f"""You are an AI tutor creating a review question for a student.
The student's current mastery level for '{review_topic}' is '{current_mastery}'.
Generate ONE challenging but fair question that will help reinforce their understanding.
The question should be based *only* on the provided context.
Use the chat history to understand what aspects of the topic the student has been working on.
Output *only* the question text, ready to present to the student."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    
    user_prompt = f"""Context for review of '{review_topic}':
<context>
{rag_context}
</context>

Generate one review question that will help reinforce the student's understanding of '{review_topic}'.
The question should be challenging but appropriate for their current mastery level of '{current_mastery}'.
Base the question *only* on the provided context.
Consider the student's recent learning context from the chat history."""

    messages.append({"role": "user", "content": user_prompt})
    return messages


def create_other_prompt(
    user_message: str,
    chat_history: List[Dict[str, str]],
    user_data: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Creates a prompt for handling general conversation and other intents.
    Uses only the last 2 user-assistant interactions for context.
    Includes user data and knowledge profile for personalized responses.
    """
    knowledge_profile = user_data.get("knowledge_profile", {})
    topics_learned = list(knowledge_profile.keys())
    topics_str = ", ".join([t.replace("_", " ") for t in topics_learned]) if topics_learned else "none yet"
    
    system_prompt = f"""You are a friendly and helpful AI tutor assistant.
The user can ask you questions and save their current knowledge level about asked topics, can test their knowledge by letting you generate quizzes or review already learnt topics based on the saved level.

Current user knowledge profile:
- Topics learned: {topics_str}
- Learning goals: {', '.join(user_data.get('learning_goals', ['none set']))}

Respond politely and concisely to greetings, thanks, confirmations, or farewells.
If the user input is unclear or asks for something outside the scope of CS/DS/AI tutoring (like jokes, personal opinions, unrelated tasks), politely state you cannot help with that specific request and offer to assist with educational topics.
Use the chat history to maintain conversation context and provide more personalized responses.
When appropriate, reference the user's learning progress and topics they've covered."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    messages.append({"role": "user", "content": user_message})
    
    return messages


def create_nlu_prompt_messages(user_input: str, intent_list: list) -> list:
    """Creates messages for NLU intent classification."""
    intent_list_str = str(intent_list)
    system_prompt = f"""You are an expert NLU system analyzing requests for an educational chatbot (CS/DS/AI topics).
Your task is to identify the user's primary intent from this exact list: {intent_list_str}.
Identify the main subject or topic mentioned. If no specific topic is relevant or the intent is 'OTHER', the topic value MUST be null.
You MUST output ONLY a single, valid JSON object containing exactly two keys: "intent" and "topic".
The JSON object structure must be exactly: {{"intent": "...", "topic": "..."}}.
Do not include ```json``` markers, explanations, apologies, or any text outside the JSON object."""

    examples = [
        ({"role": "user", "content": "what is python?"}, 
         {"role": "assistant", "content": "{\"intent\": \"GET_INFORMATION\", \"topic\": \"python\"}"}),
        ({"role": "user", "content": "Quiz me on data structures"}, 
         {"role": "assistant", "content": "{\"intent\": \"MANAGE_KNOWLEDGE\", \"topic\": \"data structures\"}"}),
        ({"role": "user", "content": "What's my level for sorting algorithms?"}, 
         {"role": "assistant", "content": "{\"intent\": \"MANAGE_KNOWLEDGE\", \"topic\": \"sorting algorithms\"}"}),
        ({"role": "user", "content": "Review my weakest topic"}, 
         {"role": "assistant", "content": "{\"intent\": \"REQUEST_REVIEW\", \"topic\": null}"}),
        ({"role": "user", "content": "thanks"}, 
         {"role": "assistant", "content": "{\"intent\": \"OTHER\", \"topic\": null}"}),
        ({"role": "user", "content": "Can you explain that again?"}, 
         {"role": "assistant", "content": "{\"intent\": \"GET_INFORMATION\", \"topic\": null}"})
    ]

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in examples:
        messages.extend([user_msg, assistant_msg])
    messages.append({"role": "user", "content": user_input})
    return messages