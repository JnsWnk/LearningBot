from datetime import timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager


# Import your modules
import schemas
import db
import security
import models
import chatbot
import config

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Loading models...")
    loaded_models = models.load_all_models()
    app.state.ml_models = loaded_models
    app.state.user_collection = db.users_collection 
    if not hasattr(db, 'client') or db.client is None:
         print("MAIN: FATAL ERROR - MongoDB connection not established in crud module.")
    yield
    print("MAIN: Application shutdown.")
    if hasattr(app.state, 'ml_models'):
        app.state.ml_models.clear()

# --- FastAPI App ---
app = FastAPI(title="CS Study Chatbot API", lifespan=lifespan)

@app.post("/register", response_model=schemas.UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(user: schemas.UserCreate):
    db_user = db.get_user_by_username(username=user.username)
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    created_user_doc = db.create_user(user=user)
    if not created_user_doc:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not create user.")
    return schemas.UserPublic.model_validate(created_user_doc)

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_doc = db.get_user_by_username(username=form_data.username)
    if not user_doc or not security.verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=1440)
    access_token = security.create_access_token(data={"sub": user_doc["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# --- Protected Routes ---

@app.get("/users/me", response_model=schemas.UserPublic)
async def read_users_me(current_user: schemas.UserPublic = Depends(security.get_current_user)):
    return current_user

@app.get("/users/knowledge", response_model=schemas.KnowledgeProfile)
async def get_user_knowledge(current_user: schemas.UserPublic = Depends(security.get_current_user)):
    user_data = db.get_user_by_id(current_user.user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return {"knowledge_profile": user_data.get("knowledge_profile", {})}

@app.get("/statistics", response_model=schemas.StatisticsResponse)
async def get_statistics():
    """Get statistics about all users' knowledge profiles."""
    try:
        # Get all users
        all_users = list(db.users_collection.find({}))
        
        # Initialize statistics
        topic_stats = {}
        level_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total_users = len(all_users)
        
        # Process each user's knowledge profile
        for user in all_users:
            knowledge_profile = user.get("knowledge_profile", {})
            for topic, data in knowledge_profile.items():
                level = data.get("level", 0)
                
                # Update topic statistics
                if topic not in topic_stats:
                    topic_stats[topic] = {
                        "total_users": 0,
                        "average_level": 0,
                        "level_distribution": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    }
                
                topic_stats[topic]["total_users"] += 1
                topic_stats[topic]["average_level"] += level
                topic_stats[topic]["level_distribution"][level] += 1
                level_distribution[level] += 1
        
        # Calculate averages and percentages
        for topic in topic_stats:
            total = topic_stats[topic]["total_users"]
            topic_stats[topic]["average_level"] /= total
            topic_stats[topic]["percentage_of_users"] = (total / total_users) * 100
            
            # Convert level distribution to percentages
            for level in topic_stats[topic]["level_distribution"]:
                topic_stats[topic]["level_distribution"][level] = (
                    topic_stats[topic]["level_distribution"][level] / total * 100
                )
        
        # Convert overall level distribution to percentages
        total_levels = sum(level_distribution.values())
        for level in level_distribution:
            level_distribution[level] = (level_distribution[level] / total_levels * 100) if total_levels > 0 else 0
        
        return {
            "topic_statistics": topic_stats,
            "overall_level_distribution": level_distribution,
            "total_users": total_users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=schemas.ChatResponse)
async def chat_endpoint(
    request: Request,
    message: schemas.ChatMessage,
    current_user: schemas.UserPublic = Depends(security.get_current_user)
):
    user_id = current_user.user_id
    user_input = message.user_input

    ml_models = request.app.state.ml_models

    if "load_error" in ml_models:
         raise HTTPException(status_code=503, detail=f"ML Models failed to load: {ml_models['load_error']}")
    if len(ml_models) == 0:
         raise HTTPException(status_code=503, detail="ML Models not available.")

    response = chatbot.process_chat_message(
        user_input=user_input,
        user_id=user_id,
        models=ml_models,
        vector_index_name=config.VECTOR_INDEX_NAME,
        use_gpt4=True
    )

    return schemas.ChatResponse(
        bot_response=response["bot_response"],
        is_quiz=response["is_quiz"],
        topic=response["topic"]
    )

@app.post("/quiz-answer", response_model=schemas.QuizResponse)
async def quiz_answer_endpoint(
    quiz_answer: schemas.QuizAnswer,
    current_user: schemas.UserPublic = Depends(security.get_current_user)
):
    user_id = current_user.user_id

    ml_models = app.state.ml_models

    if "load_error" in ml_models:
         raise HTTPException(status_code=503, detail=f"ML Models failed to load: {ml_models['load_error']}")
    if len(ml_models) == 0:
         raise HTTPException(status_code=503, detail="ML Models not available.")

    # Process the quiz answer
    response = chatbot.process_quiz(
        question=quiz_answer.question,
        answer=quiz_answer.answer,
        user_id=user_id,
        topic=quiz_answer.topic
    )

    return schemas.QuizResponse(
        bot_response=response["bot_response"],
        evaluation=schemas.QuizEvaluation(**response["evaluation"])
    )

# --- Root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Educational Chatbot API"}