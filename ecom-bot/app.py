import os
import openai
import logging
import warnings
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

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

session_id = str(uuid.uuid4())
logging.info(f"=== New session {session_id} ===")

# Создаём модель и цепочку с памятью
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2, request_timeout=15)
memory = ConversationBufferMemory(output_key="response")
conversation = ConversationChain(llm=llm, memory=memory, return_final_only=False)

# Добавим системное сообщение в контекст с указанием инфомрации из FAQ и ORDERS
faq_path = Path("data/faq.json")
orders_path = Path("data/orders.json")
if faq_path.exists():
    with open(faq_path, "r") as f:
        faq_data = json.load(f)
if orders_path.exists():
    with open(orders_path, "r") as f:
        orders_data = json.load(f)

def get_order_status(order_id):
    """Получает статус заказа по ID из orders.json"""
    order_info = orders_data.get(order_id, "Заказ с номером {order_id} не найден. Пожалуйста, проверьте правильность номера заказа.")
    return order_info

def save_session(user_input, bot_reply, total_tokens):
    """Сохраняет сессию в файл"""
    with open(f"logs/session_{session_id}.jsonl", "a") as f:
        json.dump({"dialog": f"User: {user_input}\nBot: {bot_reply}", "usage": total_tokens}, f, ensure_ascii=False)
        f.write("\n")  # Добавляем перенос строки после каждой записи


system_message = "Ты — полезный и лаконичный ассистент сервиса доставки и заказа. Ты помогаешь пользователю отвечать на вопросы. Ты отвечаешь только на вопросы о заказах, доставке и сообщениях из истории диалога. Для ответов на вопросы используй информацию из FAQ и ORDERS. Если ответ на вопрос пользователя есть в истории диалога, то ответь его. Если информации нет, ничего не придумыай сам, ответь вежливо что не можешь помочь.\nПользователи могут использовать команду /order <номер_заказа> для получения актуальной информации о статусе своих заказов.\n\n"

system_message =system_message + f"FAQ: {faq_data}\nи ORDERS: {orders_data}"
conversation.memory.chat_memory.add_message(SystemMessage(content=system_message))

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
            order_status = get_order_status(order_id)
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
        response = conversation.invoke(input=user_input)
        # Форматируем и выводим ответ
        bot_reply = response['response'].strip()
        total_tokens = response['full_generation'][0].message.response_metadata['token_usage']['total_tokens']
        print(total_tokens)
        logging.info(f"Bot: {bot_reply}. Total tokens: {total_tokens}")
        print(f"Бот: {bot_reply}")
        save_session(user_input, bot_reply, total_tokens)
    except Exception as e:
        # Логируем и выводим ошибку, продолжаем чат
        logging.error(f"Error: {e}")
        print(f"Бот: [Ошибка] {e}")
        continue
