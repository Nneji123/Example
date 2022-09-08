import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from docquery.document import ImageDocument, load_document
from docquery.ocr_reader import get_ocr_reader
from docquery.pipeline import get_pipeline
from PIL import Image, ImageDraw


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


CHECKPOINTS = {
    "LayoutLMv1 for Invoices 🧾": "impira/layoutlm-invoices",
}

PIPELINES = {}


def construct_pipeline(model):
    global PIPELINES
    if model in PIPELINES:
        return PIPELINES[model]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = get_pipeline(checkpoint=CHECKPOINTS[model], device=device)
    PIPELINES[model] = ret
    return ret


def run_pipeline(model, question, document, top_k):
    pipeline = construct_pipeline(model)
    return pipeline(question=question, **document.context, top_k=top_k)


def lift_word_boxes(document, page):
    return document.context["image"][page][1]


def expand_bbox(word_boxes):
    if len(word_boxes) == 0:
        return None

    min_x, min_y, max_x, max_y = zip(*[x[1] for x in word_boxes])
    min_x, min_y, max_x, max_y = [min(min_x), min(min_y), max(max_x), max(max_y)]
    return [min_x, min_y, max_x, max_y]


# LayoutLM boxes are normalized to 0, 1000
def normalize_bbox(box, width, height, padding=0.005):
    min_x, min_y, max_x, max_y = [c / 1000 for c in box]
    if padding != 0:
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(max_x + padding, 1)
        max_y = min(max_y + padding, 1)
    return [min_x * width, min_y * height, max_x * width, max_y * height]


FIELDS = {
    "Vendor Name": ["Vendor Name - Logo?", "Vendor Name - Address?"],
    "Vendor Address": ["Vendor Address?"],
    "Customer Name": ["Customer Name?"],
    "Customer Address": ["Customer Address?"],
    "Invoice Number": ["Invoice Number?"],
    "Invoice Date": ["Invoice Date?"],
    "Due Date": ["Due Date?"],
    "Subtotal": ["Subtotal?"],
    "Total Tax": ["Total Tax?"],
    "Invoice Total": ["Invoice Total?"],
    "Amount Due": ["Amount Due?"],
    "Payment Terms": ["Payment Terms?"],
    "Remit To Name": ["Remit To Name?"],
    "Remit To Address": ["Remit To Address?"],
}


def process_document(document, fields, model, error=None):
    if document is not None and error is None:
        json_output, table = process_fields(document, fields, model)
        return (
            document,
            fields,
            json_output,
            table,
        )
    else:
        return None


def annotate_page(prediction, pages, document):
    if prediction is not None and "word_ids" in prediction:
        image = pages[prediction["page"]]
        draw = ImageDraw.Draw(image, "RGBA")
        word_boxes = lift_word_boxes(document, prediction["page"])
        x1, y1, x2, y2 = normalize_bbox(
            expand_bbox([word_boxes[i] for i in prediction["word_ids"]]),
            image.width,
            image.height,
        )
        draw.rectangle(((x1, y1), (x2, y2)), fill=(0, 255, 0, int(0.4 * 255)))
        image.save("annotated.png")


def process_fields(document, fields, model=list(CHECKPOINTS.keys())[0]):
    pages = [x.copy().convert("RGB") for x in document.preview]

    ret = {}
    table = []

    for (field_name, questions) in fields.items():
        answers = [
            a
            for q in questions
            for a in ensure_list(run_pipeline(model, q, document, top_k=1))
            if a.get("score", 1) > 0.5
        ]
        answers.sort(key=lambda x: -x.get("score", 0) if x else 0)
        top = answers[0] if len(answers) > 0 else None
        annotate_page(top, pages, document)
        ret[field_name] = top
        table.append([field_name, top.get("answer") if top is not None else None])
        df = pd.DataFrame(table, columns=["Field", "Value"])
        # print(df)
        df.to_csv("output.csv", index=False)
    return table


def load_document_pdf(
    pdf: str = "filename.pdf", fields=FIELDS, model=list(CHECKPOINTS.keys())[0]
):

    document = load_document(pdf)
    return process_document(document, fields, model)


def load_document_image(img, fields=FIELDS, model=list(CHECKPOINTS.keys())[0]):
    document = ImageDocument(Image.fromarray(img), ocr_reader=get_ocr_reader())
    return process_document(document, fields, model)
