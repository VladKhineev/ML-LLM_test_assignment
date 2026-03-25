# Document QA

REST API для извлечения данных из документов с помощью LLM.

## Стек

- FastAPI, LangChain, FAISS, Ollama (qwen3:1.7b)

## Запуск
```bash
docker compose up --build
```

При первом запуске автоматически скачается модель qwen3:1.7b (~1 GB).

## API

### Загрузить документ
```
POST /upload
Content-Type: multipart/form-data
file: <.docx файл>

→ {"file_id": "..."}
```

### Задать вопрос
```
POST /ask
Content-Type: application/json
{"file_id": "...", "question": "Укажи предмет договора"}

→ {"question_id": "..."}
```

### Получить ответ
```
GET /answer/{question_id}

→ {"status": "done", "result": {"предмет договора": "..."}}
→ {"status": "processing"}
```

## Пример
```bash
# Загрузить файл
curl -X POST http://localhost:8000/upload -F "file=@contract.docx"

# Задать вопрос
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"file_id": "ВАШ_FILE_ID", "question": "Укажи предмет договора"}'

# Получить ответ
curl http://localhost:8000/answer/ВАШ_QUESTION_ID
```
