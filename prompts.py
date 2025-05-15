from typing import List, Dict, Optional, Any
import datetime 

def format_chat_history(chat_history: List[Dict[str, Any]], max_turns: int = 2) -> List[Dict[str, str]]:
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
You are a friendly and encouraging AI tutor. Your goal is to explain the Computer Science, Data Science, or AI topic from the 'User Request' simply and concisely (e.g., 2-4 key sentences for an initial explanation). Use the 'Retrieved Context' below to answer.
If the request is a follow-up, use the 'Chat History' to understand what it refers to.
Speak directly to the student in a supportive tone.
If the context doesn't contain the answer, politely say so (e.g., "I don't have that specific information in my current materials.").
If the user asks to summarize, provide a brief summary based ONLY on the context.

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

    system_prompt = """You are a friendly, patient, and encouraging AI tutor specializing in Computer Science, Data Science, and AI.
Your goal is to provide clear, accurate, and *concise* explanations (e.g., 2-4 main sentences for an initial explanation, and you can offer to elaborate if the student asks for more detail).
Speak directly to the student in a supportive tone (e.g., "That's a great question!", "Let's explore this concept...").
Base your answer mostly on the provided "Retrieved Context".
If the user asks a follow-up question (like "tell me more", "why?", "explain that part"), use the "Chat History" to understand what they are referring to, but still ground your new explanation in the "Retrieved Context" if relevant.
If the user asks to summarize, provide a brief summary *only* from the context, focusing on key takeaways for a student."""

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
    current_level: Any,
    rag_context: str,
    user_data: Dict[str, Any],
    chat_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:

    system_prompt = f"""You are a friendly and encouraging AI tutor creating ONE quiz question for a student about the topic: '{topic}'.
The student's current estimated understanding of this topic is '{current_level}' from a range of 0 to 5.
Your question should:
1. Use additional knowledge about the topic from a RAG databse in the context field. This is only for you and the user doesnt have this information.
2. Test a single, core concept or key piece of information from the context.
3. Be solvable by a student at level {current_level} from a range of 0-5. It should be clear, fair, and not overly complex or tricky.
4. Be concise and easy to understand.
Consider the chat history for the student's recent learning focus if it provides hints.
Output *only* the question text itself, ready to be asked. Do not add any preamble like "Here is your question:". Just the question."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    
    user_prompt = f"""Context about '{topic}':
<context>
{rag_context}
</context>

Please generate one question based on the system instructions above for {current_level} of 5 level based only on the context above.
The question should help assess and improve their understanding of the topic."""

    messages.append({"role": "user", "content": user_prompt})
    return messages


def create_request_review_prompt(
    review_topic: str,
    current_level: Any,
    rag_context: str,
    user_data: Dict[str, Any],
    chat_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:

    system_prompt = f"""You are an AI tutor creating ONE review question for a student about '{review_topic}'.
The student's current mastery for this is level {current_level} out of a range from 0 to 5. Your goal is to help reinforce their understanding.
Your question should:
1. Use additional knowledge about the topic from a RAG databse in the context field. This is only for you and the user doesnt have this information.
2. Focus on a key concept they should remember.
3. Be fair, solvable, and appropriate for solidifying understanding at their current level.
4. Be concise.
Consider the chat history for recent interactions on this topic if relevant.
Output *only* the question text. Just the question."""

    formatted_history = format_chat_history(chat_history, max_turns=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(formatted_history)
    
    user_prompt = f"""Context for review of '{review_topic}':
<context>
{rag_context}
</context>

Please generate one review question based on the system instructions above about '{review_topic}'.
The question should be appropriate for their current mastery level of {current_level} of 5."""

    messages.append({"role": "user", "content": user_prompt})
    return messages


def create_other_prompt(
    user_message: str,
    chat_history: List[Dict[str, str]],
    user_data: Dict[str, Any]
) -> List[Dict[str, str]]:
    knowledge_profile = user_data.get("knowledge_profile", {})
    topics_learned = list(knowledge_profile.keys())
    topics_str = ", ".join([t.replace("_", " ") for t in topics_learned]) if topics_learned else "none yet"
    
    system_prompt = f"""You are a friendly, patient, and supportive AI tutor assistant.
Respond politely, warmly, and *very concisely* to general conversational input like greetings, thanks, or simple confirmations.
If the user input is unclear regarding CS/DS/AI, or asks for something completely out of scope (jokes, personal opinions, complex non-educational tasks), politely state you're here to help with educational topics in Computer Science, Data Science, and AI, and ask if they have a question on those subjects.

Current user knowledge profile:
- Topics learned: {topics_str}"""

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

def create_quiz_evaluation_prompt(question: str, user_answer: str, topic: str) -> list:
    context_info = f"The question was likely based on the following topic: '{topic}'"

    system_prompt = f"""You are an encouraging and helpful AI tutor evaluating a student's answer to a quiz question about '{topic}'.
Your feedback should be supportive, clear, concise, and help the student learn. Focus on reinforcing understanding.
Speak *directly to the student* (e.g., "Your answer is good because...", "You could improve by...").

Your task is to:
1. Evaluate the student's answer for accuracy and understanding against the question (and provided context if any). Assign a score from 0 (completely incorrect/off-topic) to 5 (perfect or excellent understanding).
2. Provide a concise 'sample_solution' or ideal answer to the original question.
3. Write a personalized 'evaluation' explaining their score. If the answer is good, highlight what they did well. If it needs improvement, gently point out areas and explain the concepts clearly. Be positive and reinforcing even if the answer is incorrect. Avoid harsh language.

Return ONLY a VALID JSON OBJECT with the following structure:
{{
    "score": <integer_between_0_and_5>,
    "sample_solution": "<string_detailed_ideal_answer_to_the_original_question>",
    "evaluation": "<string_personalized_feedback_to_the_student_explaining_the_score>"
}}
Do not include ```json``` or any other text outside this JSON object."""

    user_prompt_content = f"""{context_info}
Original Question: "{question}"
Student's Answer: "{user_answer}"

Please evaluate this answer according to the system instructions and provide the response in the specified JSON format."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_content}
    ]