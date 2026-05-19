from fastapi import APIRouter, Request
from src.models import ChatRequest
from src.qa_chain import create_chain
import logging, asyncio
# from src.limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat/")
# @limiter.limit("5/minute")  # Limit to 5 requests per minute per IP
async def chat(request: ChatRequest, app_request: Request):
    logger.info(f"Received query: {request.query}")
    retriever = app_request.app.state.retriever
    chain = create_chain(retriever, session_id=request.session_id)
    response = await asyncio.to_thread(chain.invoke, {"question": request.query})
    logger.info(f"Generated response: {response}")
    return {
        "response": response["answer"], 
        "sources": [doc.page_content[:200] for doc in response["source_documents"]]
        }
