from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
import logging

from sympy import false

from untitled38 import initialize_rag, generateAnswer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agile Process Guide API",
    description="API for querying Agile process knowledge using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat-app-git-master-pabasaras-projects-9edb548c.vercel.app"],  
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        response = generateAnswer(
            RAG_LLM=rag_llm,
            chroma_collection=chroma_collection,
            query=request.prompt,
            n_results=10,
            only_response= False
        )

        # Extract sources if requested
        sources = None
        if True:
            # Get documents for sources
            results = chroma_collection.query(
                query_texts=[request.prompt],
                include=["metadatas"],
                n_results=5
            )
            logger.info(f"Reslts: {results}")
            sources = [
                f"{meta['document']} ({meta['category']})"
                for meta in results['metadatas'][0]
            ]
        logger.info(f"Sources used in response generation: {sources}")

        logger.info("Successfully generated response")
        return ChatResponse(bot=response, sources=sources)

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