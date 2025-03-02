from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
import uvicorn
import time
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_HUB_API_TOKEN = os.getenv("HUGGINGFACE_HUB_API_TOKEN")
login(token = HUGGINGFACE_HUB_API_TOKEN)
app = FastAPI(title="Serve Speculate")

# Initialize vLLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

class ModelData(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "organization"
    model: str

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    model_name = llm.model_config.model
    return ModelsResponse(
        data=[
            ModelData(
                id=model_name,
                model=llm.model_config.model
            )
        ]
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # Convert chat messages to prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens if request.max_tokens else 100,
        stop=request.stop if request.stop else [],
        n=request.n
    )
    
    # Generate response
    outputs = llm.generate([prompt], sampling_params)
    
    # Format response
    choices = []
    for idx, output in enumerate(outputs[0].outputs):
        choices.append(
            ChatChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=output.text
                ),
                finish_reason="stop"
            )
        )
    
    # Calculate token usage (approximate)
    prompt_tokens = len(prompt.split())
    completion_tokens = sum(len(choice.message.content.split()) for choice in choices)
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{outputs[0].request_id}",
        created=outputs[0].request_id,
        model=request.model,
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10050, reload=True)
