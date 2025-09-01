from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from utils.chat import get_response, get_response_stream
from models.chat import QueryRequest

router = APIRouter()


@router.post("/get-response")
def get_response_endpoint(request: QueryRequest):
    response = get_response(request.query)
    return response


@router.post("/get-response-stream")
def get_response_stream_endpoint(request: QueryRequest):
    def event_generator():
        for token in get_response_stream(request.query):
            yield f"data: {token}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
