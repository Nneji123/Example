import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,
                                  PDFSyntaxError)
from PIL import Image
from utils import load_document_image, load_document_pdf

app = FastAPI(
    title="Invoice Extractor API",
    description="""An API for extracting information from invoices""",
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
    Invoice Extractor API
    An API for extracting information from invoices
    Note: add "/redoc" to get the complete documentation.
    """
    return note


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


@app.post(
    "/invoice-pdf",
    tags=["Get Invoice from PDF"],
    description="Upload a PDF file and get the invoice information",
)
async def get_document(
    type_of_response: str = "csv",
    file: UploadFile = File(...),
):
    files = await file.read()
    # save the file
    filename = "filename.pdf"
    with open(filename, "wb+") as f:
        f.write(files)
    # open the file and return the file name
    try:
        load_document_pdf(filename)
        if os.path.exists("filename.pdf"):
            os.remove("filename.pdf")
        if type_of_response == "csv":
            return FileResponse(path="output.csv", media_type="text/csv")
        elif type_of_response == "image":
            return FileResponse(path="annotated.png", media_type="image/png")

    except (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
        ValueError,
    ) as e:
        print(e)
        if os.path.exists("filename.pdf"):
            os.remove("filename.pdf")
        if type_of_response == "csv":
            return FileResponse(path="output.csv", media_type="text/csv")
        elif type_of_response == "image":
            return FileResponse(path="annotated.png", media_type="image/png")


@app.post(
    "/invoice-image",
    tags=["Get Invoice from Image"],
    description="Upload an image of an invoice",
)
async def get_image(type_of_response: str = "csv", file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        im = Image.open("images.jpg")
        im = np.asarray(im)
        load_document_image(im)
        if os.path.exists("images.jpg"):
            os.remove("images.jpg")
        if type_of_response == "csv":
            return FileResponse(path="output.csv", media_type="text/csv")
        elif type_of_response == "image":
            return FileResponse(path="annotated.png", media_type="image/png")
    except ValueError as e:
        print(e)
        if os.path.exists("images.jpg"):
            os.remove("images.jpg")
        if type_of_response == "csv":
            return FileResponse(path="output.csv", media_type="text/csv")
        elif type_of_response == "image":
            return FileResponse(path="annotated.png", media_type="image/png")

# def remove_files():
#     import time 
#     # remove the files after 5 minutes
#     now = time.time()
#     if os.path.exists("output.csv"):
#         os.remove("output.csv")
#     if os.path.exists("annotated.png"):
#         os.remove("annotated.png")
    