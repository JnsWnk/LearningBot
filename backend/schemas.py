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

class ChatResponse(BaseModel):
    bot_response: str