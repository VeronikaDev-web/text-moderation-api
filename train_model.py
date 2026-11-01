# train_model.py
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

X_train = [
    # Спам / Реклама
    "купить дешево скидка акция бесплатно рассылка",
    "выигрыш бонус перейдите по ссылке срочно акция",
    "распродажа криптовалюта легкий заработок доход",
    # Поддержка / Вопросы
    "подскажите как настроить оплату в личном кабинете",
    "не могу войти в аккаунт забыл пароль помогите",
    "как изменить номер телефона в профиле",
    "подскажите график работы вашей поддержки",
    # Отзывы / Похвала
    "спасибо огромное за отличный сервис и быструю доставку",
    "очень понравилось качество товара буду зазывать еще",
    "прекрасный персонал все супер спасибо",
    # Жалобы
    "ужасный сервис товар пришел сломанным требую возврат денег",
    "почему так долго идет доставка отмените заказ",
    "курьер хамил качество ужасное верните средства",
]

y_train = [
    "spam",
    "spam",
    "spam",
    "support",
    "support",
    "support",
    "support",
    "feedback",
    "feedback",
    "feedback",
    "complaint",
    "complaint",
    "complaint",
]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/text_classifier.pkl")
print("Продвинутая модель успешно обучена и сохранена!")