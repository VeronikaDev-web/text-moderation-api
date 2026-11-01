import streamlit as st
import requests

# Настройка страницы браузера
st.set_page_config(page_title="Мой ИИ-Чат", page_icon="🤖", layout="centered")

st.title("💬 Чат с моделью через собственное API")
st.write("Этот интерфейс отправляет ваши сообщения на локальный FastAPI-сервер, где их анализирует модель.")

# Проверяем, запущено ли наше API, и сохраняем историю сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Выводим историю сообщений на экран при перезагрузке
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода текста внизу страницы
if prompt := st.chat_input("Введите сообщение для проверки (например: купить скидку)..."):
    
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Отправляем запрос в наше FastAPI приложение
    with st.chat_message("assistant"):
        with st.spinner("Модель думает..."):
            try:
                # Адрес нашего эндпоинта FastAPI
                api_url = "http://127.0.0.1:8000/predict-spam"
                
                # Формируем JSON-запрос
                payload = {"message": prompt}
                
                # Делаем POST-запрос к нашему API
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction")
                    confidence = data.get("confidence")
                    
                    # Красиво форматируем ответ от модели
                    if prediction == "spam":
                        reply = f"🚨 **Вердикт:** Это СПАМ!\n\n*Уверенность модели:* `{confidence * 100:.1f}%`"
                    else:
                        reply = f"✅ **Вердикт:** Нормальное сообщение (не спам).\n\n*Уверенность модели:* `{confidence * 100:.1f}%`"
                else:
                    reply = f"⚠️ Ошибка сервера API: код {response.status_code}"
                    
            except requests.exceptions.ConnectionError:
                reply = "❌ **Ошибка подключения:** Не удалось связаться с FastAPI. Убедитесь, что сервер запущен (`python -m uvicorn main:app --reload`)."

            st.markdown(reply)
            
            # Добавляем ответ ассистента в историю
            st.session_state.messages.append({"role": "assistant", "content": reply})