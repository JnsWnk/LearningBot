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
    users_collection.create_index("username", unique=True)
    print("CRUD: MongoDB connection established and user index ensured.")
except Exception as e:
    print(f"CRUD: Error connecting to MongoDB: {e}")

def get_user_by_username(username: str): # -> Optional[schemas.UserInDBBase]
    user_doc = users_collection.find_one({"username": username})
    if user_doc:
        user_doc['_id'] = str(user_doc['_id'])
    return user_doc 

def create_user(user: schemas.UserCreate): # -> Optional[dict]
    hashed_password = security.get_password_hash(user.password)
    user_doc = {
        "username": user.username,
        "hashed_password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "knowledge_profile": {},
        "learning_goals": []
    }
    try:
        insert_result = users_collection.insert_one(user_doc)
        created_doc = users_collection.find_one({"_id": insert_result.inserted_id})
        if created_doc:
            created_doc['_id'] = str(created_doc['_id'])
        return created_doc
    except DuplicateKeyError:
        print(f"Username '{user.username}' already exists.")
        return None
    except Exception as e:
        print(f"Error creating user: {e}")
        return None
    
def update_user_knowledge(user_id: str, concept: str, mastery_level = None, confidence = None):
    if not user_id or not concept:
        print("Error: user_id and concept are required for update_user_knowledge.")
        return None

    try:
        # Convert user_id string back to ObjectId for querying
        obj_user_id = ObjectId(user_id)
    except Exception as e:
        print(f"Error: Invalid user_id format '{user_id}': {e}")
        return None

    update_filter = {"_id": obj_user_id}
    update_fields = {}
    changes_made = False

    # Prepare fields to update using dot notation
    if mastery_level is not None:
        update_fields[f"knowledge_profile.{concept}.mastery"] = mastery_level
        changes_made = True
    if confidence is not None:
        update_fields[f"knowledge_profile.{concept}.confidence"] = confidence
        changes_made = True

    # Only update if there are actual changes requested
    if changes_made:
        # Always update the last_reviewed timestamp when updating mastery/confidence
        update_fields[f"knowledge_profile.{concept}.last_reviewed"] = datetime.datetime.now(datetime.timezone.utc)
        update_operation = {"$set": update_fields}

        try:
            result = users_collection.update_one(update_filter, update_operation)
            if result.matched_count == 0:
                print(f"Warning: No user found with _id '{user_id}' to update knowledge.")
                return None
            print(f"Updated knowledge for user '{user_id}', concept '{concept}'. Matched: {result.matched_count}, Modified: {result.modified_count}")
            return result # Returns UpdateResult object
        except Exception as e:
            print(f"Error updating user knowledge in DB for user '{user_id}', concept '{concept}': {e}")
            return None
    else:
        print(f"No updates provided for user '{user_id}', concept '{concept}'.")
        return None