from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import datetime
from bson.objectid import ObjectId

import schemas
import security
import config

try:
    client = MongoClient(config.MONGO_CONNECTION_STRING)
    db = client[config.DB_NAME]
    users_collection = db[config.USER_COLLECTION]
    docs_collection = db[config.DOCS_COLLECTION]
    users_collection.create_index("username", unique=True)
    print("CRUD: MongoDB connection established and user index ensured.")
except Exception as e:
    print(f"CRUD: Error connecting to MongoDB: {e}")


def get_user_by_username(username: str):
    user_doc = users_collection.find_one({"username": username})
    if user_doc:
        user_doc['_id'] = str(user_doc['_id'])
    return user_doc


def create_user(user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    user_doc = {
        "username": user.username,
        "hashed_password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "knowledge_profile": {},
        "chat_history": []
    }
    try:
        insert_result = users_collection.insert_one(user_doc)
        created_doc = users_collection.find_one(
            {"_id": insert_result.inserted_id})
        if created_doc:
            created_doc['_id'] = str(created_doc['_id'])
        return created_doc
    except DuplicateKeyError:
        print(f"Username '{user.username}' already exists.")
        return None
    except Exception as e:
        print(f"Error creating user: {e}")
        return None


def update_user_knowledge(user_id: str, concept: str, new_level: int = None):
    if not user_id or not concept:
        print("Error: user_id and concept are required for update_user_knowledge.")
        return None

    try:
        obj_user_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error: Invalid user_id format '{user_id}': {e}")
        return None

    current_time = datetime.datetime.now(datetime.timezone.utc)

    # Get current knowledge profile
    user = users_collection.find_one({"_id": obj_user_id})
    if not user:
        print(f"Warning: No user found with _id '{user_id}'")
        return None

    # Get current level or default to 1 if not exists
    current_level = user.get("knowledge_profile", {}).get(
        concept, {}).get("level", 1)

    # Update level if provided
    if new_level is not None:
        # Ensure level stays within 0-5 range
        current_level = max(0, min(5, new_level))

    # Update the knowledge profile
    update_operation = {
        "$set": {
            f"knowledge_profile.{concept}": {
                "level": current_level,
                "last_updated": current_time
            }
        }
    }

    try:
        result = users_collection.update_one(
            {"_id": obj_user_id},
            update_operation
        )
        if result.matched_count == 0:
            print(
                f"Warning: No user found with _id '{user_id}' to update knowledge.")
            return None
        print(
            f"Updated knowledge for user '{user_id}', concept '{concept}'. Level: {current_level}")
        return result
    except Exception as e:
        print(
            f"Error updating user knowledge in DB for user '{user_id}', concept '{concept}': {e}")
        return None


def get_topic_to_review(user_id: str) -> str:
    try:
        obj_user_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error: Invalid user_id format '{user_id}': {e}")
        return None

    try:
        user = users_collection.find_one({"_id": obj_user_id})
        if not user or not user.get("knowledge_profile"):
            return None

        # Find topic with oldest last_updated date
        oldest_topic = min(
            user["knowledge_profile"].items(),
            key=lambda x: x[1].get("last_updated", datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc))
        )[0]

        return oldest_topic
    except Exception as e:
        print(f"Error getting topic to review for user '{user_id}': {e}")
        return None


def save_chat_history(user_id: str, prompt: str, answer: str):
    if not user_id or not prompt or not answer:
        print("Error: user_id, prompt, and answer are required for save_chat_history.")
        return None

    try:
        obj_user_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error: Invalid user_id format '{user_id}': {e}")
        return None

    chat_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "prompt": prompt,
        "answer": answer
    }

    try:
        result = users_collection.update_one(
            {"_id": obj_user_id},
            {
                "$push": {
                    "chat_history": {
                        "$each": [chat_entry],
                        "$slice": -10
                    }
                }
            }
        )
        if result.matched_count == 0:
            print(
                f"Warning: No user found with _id '{user_id}' to save chat history.")
            return None
        return result
    except Exception as e:
        print(f"Error saving chat history for user '{user_id}': {e}")
        return None


def get_chat_history(user_id: str, limit: int = 1) -> list:
    if not user_id:
        print("Error: user_id is required for get_chat_history.")
        return []

    try:
        obj_user_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error: Invalid user_id format '{user_id}': {e}")
        return []

    try:
        user = users_collection.find_one(
            {"_id": obj_user_id},
            {"chat_history": {"$slice": -limit}}
        )

        if not user or "chat_history" not in user:
            return []

        return user["chat_history"]
    except Exception as e:
        print(f"Error retrieving chat history for user '{user_id}': {e}")
        return []


def get_user_by_id(user_id: str):
    if not user_id:
        print("Error: user_id is required for get_user_by_id.")
        return None

    try:
        obj_user_id = ObjectId(user_id)
        user_doc = users_collection.find_one({"_id": obj_user_id})
        if user_doc:
            user_doc['_id'] = str(user_doc['_id'])
        return user_doc
    except Exception as e:
        print(f"Error retrieving user by ID '{user_id}': {e}")
        return None
