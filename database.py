import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./saas_database.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Client(Base):
  __tablename__ = "clients"

  id = Column(Integer, primary_key=True, index=True)
  api_key = Column(String, unique=True, index=True)
  name = Column(String)
  tier = Column(
      String, default="free"
  )  # тариф: "free", "pro", "enterprise"
  requests_count = Column(Integer, default=0)  # счетчик запросов


class RequestLog(Base):
  __tablename__ = "request_logs"

  id = Column(Integer, primary_key=True, index=True)
  api_key = Column(String, index=True)
  text = Column(String)
  category = Column(String)
  confidence = Column(Float)
  toxicity = Column(Float)
  sentiment = Column(String)
  timestamp = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
  Base.metadata.create_all(bind=engine)
  # Создадим тестового клиента при первом запуске, если его нет
  db = SessionLocal()
  existing_client = db.query(Client).filter(Client.api_key == "client_key").first()
  if not existing_client:
    test_client = Client(
        api_key="client_key", name="Test User", tier="free", requests_count=0
    )
    admin_client = Client(
        api_key="admin_key", name="Admin User", tier="enterprise", requests_count=0
    )
    db.add(test_client)
    db.add(admin_client)
    db.commit()
  db.close()