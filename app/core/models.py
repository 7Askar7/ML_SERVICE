from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import datetime
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    credits = Column(Float, default=50.0)
    lang = Column(String, default="ru")

class UserMessage(Base):
    __tablename__ = "user_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String) 
    content = Column(String)
    lang = Column(String, default="ru")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User")