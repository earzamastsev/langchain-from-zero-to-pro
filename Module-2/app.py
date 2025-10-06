import os
import openai
import logging
import warnings
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from src.brand_chain import ChatBot

# Отключаем предупреждения LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Инициализация: загрузка переменных, настройка логов, создание бота
load_dotenv()  # загружаем .env, чтобы получить ключи
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

# Настраиваем логирование в файл
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/chat_session.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Создаём экземпляр чат-бота
chatbot = ChatBot()

print("Чат-бот запущен! Можете задавать вопросы. Для выхода введите 'выход'.\n")

# 2. Главный цикл диалога
while True:
    try:
        user_input = input("Вы: ")
    except KeyboardInterrupt:
        print("\n[Прерывание пользователем]")
        logging.info("User initiated exit by keyboard interrupt. Session ended.")
        break
    # Удаляем лишние пробелы
    user_input = user_input.strip()
    
    if user_input == "":
        continue  # пустой ввод - пропускаем
    
    # Проверка команды /order <id>
    if user_input.startswith('/order '):
        order_id = user_input[7:].strip()  # убираем '/order ' и пробелы
        if order_id:
            order_status = chatbot.get_order_status(order_id)
            user_input = f"Пользователь интересуется статусом заказа {order_id}. Подготовь ответ на вопрос пользователя используя информацию по статусу заказа: {order_status}"
        else:
            print("Бот: Пожалуйста, укажите номер заказа после команды /order")
            continue
    
    # Проверка команды выхода
    if user_input.lower() in ("выход", "quit", "exit"):
        print("Бот: До свидания!")
        logging.info("User initiated exit by command. Session ended.")
        break

    logging.info(f"User: {user_input}")
    # Генерация ответа с обработкой ошибок
    try:
        # Используем метод chat из класса ChatBot
        bot_reply = chatbot.chat(user_input)
        logging.info(f"Bot: {bot_reply}")
        print(f"Бот: {bot_reply}")
        chatbot.save_session(user_input, bot_reply)
    except Exception as e:
        # Логируем и выводим ошибку, продолжаем чат
        logging.error(f"Error: {e}")
        print(f"Бот: [Ошибка] {e}")
        continue
