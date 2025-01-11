from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import logging
from fastapi import Request
from fastapi.responses import StreamingResponse 
import asyncio

from sympy import false

from untitled38 import initialize_rag, generateAnswer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Trading guide API",
    description="API for querying Crypto Trading knowledge using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
try:
    chroma_collection, rag_llm = initialize_rag()
    logger.info("RAG components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG components: {str(e)}")
    raise


class ChatRequest(BaseModel):
    prompt: str
    show_sources: Optional[bool] = False


class ChatResponse(BaseModel):
    bot: str
    sources: Optional[list] = None


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Agile Process Guide API is running"}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    try:
        auth_header = http_request.headers.get("Authorization", "")
        jwt = auth_header.replace("Bearer ", "")
        if not jwt:
            raise HTTPException(
                status_code=401,
                detail="Missing JWT in Authorization header"
            )

        logger.info(f"Received JWT: {jwt[:10]}... (truncated)")

        # Define an async generator to stream the response
        async def response_generator():
            response = ""
            
            # Example of streaming response generation
            for i, chunk in enumerate(generateAnswer(RAG_LLM=rag_llm,
            chroma_collection=chroma_collection,
            query=request.prompt,
            n_results=10,
            jwt_token=jwt)):
                
                response += chunk
                yield chunk
                await asyncio.sleep(0.1)  # Simulate delay (optional)

            logger.info(f"Final response: {response}")

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the API server")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the API server")


if __name__ == "__main__":
    try:
        logger.info("Starting the server...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise