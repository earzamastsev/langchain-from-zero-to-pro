import os
import openai
import logging
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1. Инициализация: загрузка переменных, настройка логов, создание бота
load_dotenv()  # загружаем .env, чтобы получить ключи
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.essayai.ru/v1")

# Настраиваем логирование в файл
logging.basicConfig(
    filename="chat_session.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("=== New session ===")

# Создаём модель и цепочку с памятью
llm = ChatOpenAI(model_name="gpt-5", temperature=0, request_timeout=15)
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

# (Опционально) Добавим системное сообщение в контекст, если нужно
# from langchain.schema import SystemMessage
# conversation.memory.chat_memory.add_message(SystemMessage(content="Ты — полезный и лаконичный ассистент."))

print("Чат-бот запущен! Можете задавать вопросы. Для выхода введите 'выход'.\n")

# 2. Главный цикл диалога
while True:
    try:
        user_input = input("Вы: ")
    except KeyboardInterrupt:
        print("\n[Прерывание пользователем]")
        break
    # Удаляем лишние пробелы
    user_input = user_input.strip()
    if user_input == "":
        continue  # пустой ввод - пропускаем
    # Проверка команды выхода
    if user_input.lower() in ("выход", "quit", "exit"):
        print("Бот: До свидания!")
        logging.info("User initiated exit. Session ended.")
        break

    logging.info(f"User: {user_input}")
    # Генерация ответа с обработкой ошибок
    try:
        bot_reply = conversation.predict(input=user_input)
    except Exception as e:
        # Логируем и выводим ошибку, продолжаем чат
        logging.error(f"Error: {e}")
        print(f"Бот: [Ошибка] {e}")
        continue

    # Форматируем и выводим ответ
    bot_reply = bot_reply.strip()
    logging.info(f"Bot: {bot_reply}")
    print(f"Бот: {bot_reply}")
