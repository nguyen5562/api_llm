from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import system, file, embedding, chat

app = FastAPI()

# Bật CORS cho mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi origin
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức
    allow_headers=["*"],  # Cho phép tất cả headers
)

# Include các router
app.include_router(chat.router)
app.include_router(system.router)
app.include_router(file.router)
app.include_router(embedding.router)
