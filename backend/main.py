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
async def read_users_me(current_user: dict = Depends(security.get_current_user)):
    return current_user


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

    bot_answer = chatbot.process_chat_message(
        user_input=user_input,
        user_id=user_id,
        models=ml_models,
        vector_index_name=config.VECTOR_INDEX_NAME,
        use_gpt4=True
    )

    return schemas.ChatResponse(bot_response=bot_answer)


# --- Root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Educational Chatbot API"}