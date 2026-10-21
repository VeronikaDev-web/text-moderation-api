# main.py
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import os
from dotenv import load_dotenv

# Загружаем секреты из локального файла .env (если он есть)
load_dotenv()

app = FastAPI(
    title="Text Analytics & Moderation API",
    description="Коммерческий API-сервис классификации текстов, модерации и аналитики.",
    version="1.0.0"
)

# Настройка API-ключей
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Безопасно подтягиваем ключи из переменных окружения
VALID_API_KEYS = {
    os.getenv("CLIENT_KEY", "demo_key_12345"): "Клиент Тестовый",
    os.getenv("ADMIN_KEY", "verona_secret_key_999"): "Администратор"
}

def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403, 
            detail="Недействительный или отсутствующий API-ключ. Доступ запрещен."
        )
    return api_key

# Загрузка модели
MODEL_PATH = "text_classifier.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Модель успешно загружена!")
    else:
        print("Файл модели не найден. Запустите train_model.py")

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/v1/analyze", dependencies=[Depends(verify_api_key)])
def analyze_text(payload: AnalyzeRequest):
    """
    Эндпоинт комплексного анализа текста.
    Требует передачу заголовка X-API-Key.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Текст сообщения не может быть пустым.")
    
    # 1. Анализ моделью ML
    category = model.predict([payload.text])[0]
    probs = model.predict_proba([payload.text])[0]
    confidence = float(max(probs))
    
    # 2. Дополнительная аналитика
    words = payload.text.strip().split()
    char_count = len(payload.text)
    
    # Определение флага модерации
    is_flagged = True if category in ["spam", "complaint"] else False
    
    return {
        "status": "success",
        "result": {
            "category": category,              # 'spam', 'support', 'feedback', 'complaint'
            "confidence": round(confidence, 4),# Уверенность модели
            "is_flagged": is_flagged,          # Требует ли внимания модератора
            "metrics": {
                "word_count": len(words),
                "character_count": char_count
            }
        }
    }