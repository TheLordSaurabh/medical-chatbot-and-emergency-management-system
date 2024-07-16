import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

# Retrieve DATABASE_URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    conversations = relationship("Conversation", back_populates="owner")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    chats = relationship("Chat", back_populates="conversation")
    owner = relationship("User", back_populates="conversations")
    summary_record = relationship("ConversationSummary", uselist=False, back_populates="conversation")
    predicted_disease = Column(String)

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    chat_id = Column(Integer)
    question = Column(Text)
    response = Column(Text)
    conversation = relationship("Conversation", back_populates="chats")

class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), unique=True)
    summary = Column(Text)
    conversation = relationship("Conversation", back_populates="summary_record")

Base.metadata.create_all(bind=engine)

# FastAPI setup
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    chat_id: int
    conversational_memory: str
    question: str

class ChatResponse(BaseModel):
    conversation_id: int
    chat_id: int
    conversational_memory: str
    response: str

class EndChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    conversational_memory: str
    question: str

class PredictedDiseaseResponse(BaseModel):
    conversation_id: int
    predicted_disease: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load the model and tokenizer
model_name = "sg-server/meta-llama-3-8b-instruct-disease-prediction-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Instruction for the medical chatbot
instruction = ("You're an Advanced Medical Chatbot - Chatbot Dr. MCX, so assume yourself as a Healthcare Professional "
               "whose job is to listen to the Patient's query or symptoms and predict the underlying Health Problem the patient is suffering. "
               "Carefully listen to the Patient's query or symptoms and output only the Identified Health Problem in the following format: "
               "Identified Health Problem : [Health Problem Identified by you].")

simple_prompt_template = PromptTemplate.from_template("{input_text}")
disease_prediction_prompt_template = PromptTemplate.from_template(
    f"{instruction}\n\nPatient's query or symptoms : {{input_text}}"
)

global_summary = ""

class CustomConversationSummaryBufferMemory(ConversationSummaryBufferMemory):
    def __init__(self, global_summary):
        super().__init__()
        self.global_summary = global_summary

    def add_to_memory(self, context, question, answer):
        super().add_to_memory(context, question, answer)
        self.global_summary += f"Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n"
        return self.global_summary

memory = CustomConversationSummaryBufferMemory(global_summary)

llm_chain_simple = LLMChain(
    model=model,
    tokenizer=tokenizer,
    prompt_template=simple_prompt_template,
    memory=memory
)

llm_chain_disease_prediction = LLMChain(
    model=model,
    tokenizer=tokenizer,
    prompt_template=disease_prediction_prompt_template,
    memory=memory
)

def medical_chatbot_simple(context, input_text):
    response = llm_chain_simple.run(context=context, input_text=input_text)
    global_summary = memory.add_to_memory(context=context, question=input_text, answer=response)
    return response, global_summary

def medical_chatbot_disease_prediction(context, input_text):
    response = llm_chain_disease_prediction.run(context=context, input_text=input_text)
    global_summary = memory.add_to_memory(context=context, question=input_text, answer=response)
    return response, global_summary

@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, db: Session = Depends(get_db)):
    user_id = chat_request.user_id
    conversation_id = chat_request.conversation_id
    chat_id = chat_request.chat_id
    conversational_memory = chat_request.conversational_memory
    question = chat_request.question

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        conversation_id = conversation.id

    context = conversational_memory
    response, global_summary = medical_chatbot_simple(context, question)
    
    summary_record = db.query(ConversationSummary).filter(ConversationSummary.conversation_id == conversation_id).first()
    if not summary_record:
        summary_record = ConversationSummary(conversation_id=conversation.id, summary=global_summary)
    else:
        summary_record.summary = global_summary
    db.add(summary_record)
    db.commit()

    chat = Chat(conversation_id=conversation.id, chat_id=chat_id, question=question, response=response)
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return ChatResponse(
        conversation_id=conversation_id,
        chat_id=chat.chat_id,
        conversational_memory=global_summary,
        response=response
    )

@app.post("/end-chat", response_model=PredictedDiseaseResponse)
def end_chat(end_chat_request: EndChatRequest, db: Session = Depends(get_db)):
    user_id = end_chat_request.user_id
    conversation_id = end_chat_request.conversation_id
    conversational_memory = end_chat_request.conversational_memory
    question = end_chat_request.question

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    context = conversational_memory
    response, global_summary = medical_chatbot_disease_prediction(context, question)
    
    conversation.predicted_disease = response
    db.add(conversation)
    db.commit()

    return PredictedDiseaseResponse(
        conversation_id=conversation_id,
        predicted_disease=response
    )

@app.get("/predicted-disease/{conversation_id}", response_model=PredictedDiseaseResponse)
def get_predicted_disease(conversation_id: int, db: Session = Depends(get_db)):
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not conversation.predicted_disease:
        raise HTTPException(status_code=404, detail="Predicted disease not found for this conversation")

    return PredictedDiseaseResponse(
        conversation_id=conversation_id,
        predicted_disease=conversation.predicted_disease
    )

@app.get("/conversation-summary/{conversation_id}", response_model=ConversationSummaryResponse)
def get_conversation_summary(conversation_id: int, db: Session = Depends(get_db)):
    summary_record = db.query(ConversationSummary).filter(ConversationSummary.conversation_id == conversation_id).first()
    if not summary_record:
        raise HTTPException(status_code=404, detail="Conversation summary not found")

    return ConversationSummaryResponse(
        conversation_id=conversation_id,
        summary=summary_record.summary
    )

if __name__ == "__server__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
