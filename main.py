from datetime import datetime
import os
from database import Client, RequestLog, SessionLocal, init_db
from fastapi import Depends, FastAPI, Header, HTTPException, status
import joblib
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Инициализируем БД при старте
init_db()

app = FastAPI(
    title="Enterprise Text Analytics & Moderation API",
    version="2.0.0",
    description=(
        "Коммерческий SaaS API с многоуровневым анализом, БД и лимитами."
    ),
)

# Загрузка модели
MODEL_PATH = "models/model.pkl"
try:
  model = joblib.load(MODEL_PATH)
except Exception:
  model = None


# Зависимость для получения сессии БД
def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()


# Функция аутентификации через базу данных и проверки лимитов
def verify_api_key(
    x_api_key: str = Header(..., description="Ваш SaaS API ключ"),
    db: Session = Depends(get_db),
):
  client = db.query(Client).filter(Client.api_key == x_api_key).first()
  if not client:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный или отсутствующий API ключ",
    )

  # Лимиты запросов в зависимости от тарифа (Шаг 3)
  limits = {"free": 5, "pro": 100, "enterprise": 10000}

  max_limit = limits.get(client.tier, 5)
  if client.requests_count >= max_limit:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Превышен лимит запросов для вашего тарифа ({client.tier})."
            f" Лимит: {max_limit}"
        ),
    )

  return client


class AnalyzeRequest(BaseModel):
  text: str


@app.post("/v1/analyze", tags=["Moderation"])
def analyze_text(
    payload: AnalyzeRequest,
    client: Client = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Модель машинного обучения не найдена на сервере"
    )

  # Проводим предсказание моделью
  prediction = model.predict([payload.text])[0]
  probabilities = model.predict_proba([payload.text])[0]
  confidence = float(max(probabilities))

  # Эмулируем расширенный анализ (Этап 1: Токсичность и Тональность)
  toxicity_score = (
      0.95
      if "идиот" in payload.text.lower() or "ненавижу" in payload.text.lower()
      else 0.05
  )
  sentiment = (
      "negative"
      if toxicity_score > 0.5
      else ("positive" if "спасибо" in payload.text.lower() else "neutral")
  )

  # Увеличиваем счетчик запросов клиента в БД
  client.requests_count += 1

  # Логируем запрос в базу данных (Этап 4)
  log_entry = RequestLog(
      api_key=client.api_key,
      text=payload.text,
      category=prediction,
      confidence=confidence,
      toxicity=toxicity_score,
      sentiment=sentiment,
  )
  db.add(log_entry)
  db.commit()

  return {
      "status": "success",
      "client": client.name,
      "tier": client.tier,
      "remaining_quota": client.requests_count,
      "result": {
          "category": prediction,
          "confidence": round(confidence, 4),
          "toxicity_score": toxicity_score,
          "sentiment": sentiment,
          "is_flagged": prediction == "spam" or toxicity_score > 0.5,
          "metrics": {
              "word_count": len(payload.text.split()),
              "character_count": len(payload.text),
          },
      },
  }


@app.get("/v1/stats", tags=["SaaS Analytics"])
def get_stats(client: Client = Depends(verify_api_key), db: Session = Depends(get_db)):
  """Эндпоинт для клиентов: просмотр своей статистики и истории запросов"""
  logs = (
      db.query(RequestLog)
      .filter(RequestLog.api_key == client.api_key)
      .order_by(RequestLog.timestamp.desc())
      .limit(10)
      .all()
  )

  return {
      "client_name": client.name,
      "tier": client.tier,
      "total_requests_made": client.requests_count,
      "recent_logs": [
          {
              "text": log.text,
              "category": log.category,
              "toxicity": log.toxicity,
              "sentiment": log.sentiment,
              "time": log.timestamp,
          }
          for log in logs
      ],
  }