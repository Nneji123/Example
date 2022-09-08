from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from utils import predict
import os

app = FastAPI(
    title="Comment Classifier",
    description="""An API for classifying comments.""",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Comment Classification API
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class Comment(BaseModel):
    text: str


# endpoint for just enhancing the image
@app.post("/comment", tags=['Comment Toxicity Classifier'])
async def get_comment(data: Comment):
    comment_text = predict(data.text)
    return comment_text
