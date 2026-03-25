from contextlib import asynccontextmanager
import asyncio
import uuid

import httpx
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

from src.parser import parse_docx
from src.rag import build_index, retrieve, generate_answer, OLLAMA_HOST, MODEL_NAME
from src.storage import files, questions, answers


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            await client.post(f"{OLLAMA_HOST}/api/pull", json={"name": MODEL_NAME})
        except Exception as e:
            print(f"[warn] не удалось скачать модель: {e}")
    yield


app = FastAPI(title="Document QA", lifespan=lifespan)


class AskRequest(BaseModel):
    file_id: str
    question: str


@app.post("/upload")
async def upload(file: UploadFile):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Принимаются только .docx файлы")

    file_id = str(uuid.uuid4())
    files[file_id] = await file.read()
    return {"file_id": file_id}


@app.post("/ask")
async def ask(data: AskRequest):
    if data.file_id not in files:
        raise HTTPException(status_code=404, detail="Файл не найден")

    question_id = str(uuid.uuid4())
    questions[question_id] = data.model_dump()
    answers[question_id] = {"status": "processing"}
    asyncio.create_task(process_question(question_id))
    return {"question_id": question_id}


@app.get("/answer/{question_id}")
def get_answer(question_id: str):
    if question_id not in answers:
        raise HTTPException(status_code=404, detail="Вопрос не найден")
    return answers[question_id]


async def process_question(question_id: str):
    data = questions[question_id]
    try:
        text = parse_docx(files[data["file_id"]])
        index = build_index(text)
        docs = retrieve(index, data["question"], text)
        result = generate_answer(data["question"], docs)
        answers[question_id] = {"status": "done", "result": result}
    except Exception as e:
        answers[question_id] = {"status": "error", "detail": str(e)}
