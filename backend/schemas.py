from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any

# --- User Schemas ---
class UserBase(BaseModel):
    username: str = Field(..., min_length=3)

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserInDBBase(UserBase):
    user_id: str = Field(alias="_id") # Use alias for MongoDB's _id
    hashed_password: str

    class Config:
        populate_by_name = True # Allow using '_id' when creating instance
        from_attributes = True # Allow creating from ORM/dict objects

class UserPublic(UserBase):
     user_id: str = Field(alias="_id")
     class Config:
        populate_by_name = True
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatMessage(BaseModel):
    user_input: str
    use_gpt4: bool

class ChatResponse(BaseModel):
    bot_response: str
    is_quiz: bool = False
    topic: str

class QuizAnswer(BaseModel):
    answer: str
    message_id: str
    question: str
    topic: str

class QuizEvaluation(BaseModel):
    score: int
    sample_solution: str
    evaluation: str

class QuizResponse(BaseModel):
    bot_response: str
    evaluation: QuizEvaluation

class KnowledgeProfile(BaseModel):
    knowledge_profile: Dict[str, Dict[str, Any]]

class TopicStatistics(BaseModel):
    total_users: int
    average_level: float
    percentage_of_users: float
    level_distribution: Dict[int, float]

class StatisticsResponse(BaseModel):
    topic_statistics: Dict[str, TopicStatistics]
    overall_level_distribution: Dict[int, float]
    total_users: int