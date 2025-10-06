import yaml
import os
import json
import uuid
import logging
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage


def load_style_guide() -> dict:
    """
    Загружает данные из файла style_guide.yaml и возвращает словарь.
    
    Returns:
        dict: Словарь с данными из style_guide.yaml
    """
    current_dir = Path(__file__).parent
    style_guide_path = current_dir.parent / "data" / "style_guide.yaml"
    
    try:
        with open(style_guide_path, 'r', encoding='utf-8') as file:
            style_guide = yaml.safe_load(file)
        return style_guide
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл style_guide.yaml не найден по пути: {style_guide_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка при парсинге YAML файла: {e}")
    except Exception as e:
        raise Exception(f"Неожиданная ошибка при загрузке файла: {e}")


def load_prompts() -> dict:
    """
    Загружает данные из файла prompts.yaml и возвращает словарь.
    
    Returns:
        dict: Словарь с данными из prompts.yaml
    """
    # Получаем путь к файлу prompts.yaml относительно текущего файла
    current_dir = Path(__file__).parent
    prompts_path = current_dir.parent / "data" / "prompts.yaml"
    
    try:
        with open(prompts_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл prompts.yaml не найден по пути: {prompts_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка при парсинге YAML файла: {e}")
    except Exception as e:
        raise Exception(f"Неожиданная ошибка при загрузке файла: {e}")


def create_system_prompt_template(prompt_version: str = "current") -> SystemMessagePromptTemplate:
    """
    Создает шаблон системного промпта на основе данных из prompts.yaml.
    
    Args:
        prompt_version (str): Версия промпта для использования. По умолчанию "current".
                             Если "current", используется версия из поля current.
    
    Returns:
        SystemMessagePromptTemplate: Объект SystemMessagePromptTemplate из LangChain
    """
    # Загружаем данные промптов
    prompts_data = load_prompts()
    style_guide = load_style_guide()

    # Создаем словарь переменных для промпта
    prompt_variables = {
        "brand": style_guide.get("brand", "Shoply"),
        "persona": style_guide.get("tone", {}).get("persona", "вежливый, деловой, дружелюбный"),
        "sentences_max": style_guide.get("tone", {}).get("sentences_max", 3),
        "bullets": style_guide.get("tone", {}).get("bullets", True),
        "avoid": ", ".join(style_guide.get("tone", {}).get("avoid", [])),
        "must_include": ", ".join(style_guide.get("tone", {}).get("must_include", [])),
        "fallback": style_guide.get("fallback", {}).get("no_data", "У меня нет точной информации.")
    }
    
    # Определяем версию промпта
    if prompt_version == "current":
        version = prompts_data["prompts"]["current"]
    else:
        version = prompt_version
    
    # Получаем данные для выбранной версии
    if version not in prompts_data["prompts"]:
        raise ValueError(f"Версия промпта '{version}' не найдена в файле prompts.yaml")
    
    prompt_config = prompts_data["prompts"][version]
    
    # Проверяем наличие системного промпта
    if "system" not in prompt_config:
        raise ValueError(f"Системный промпт не найден для версии '{version}'")
    
    # Создаем и возвращаем шаблон системного сообщения с переменными
    return SystemMessagePromptTemplate.from_template(prompt_config["system"], partial_variables=prompt_variables)


def create_user_prompt_template(prompt_version: str = "current") -> HumanMessagePromptTemplate:
    """
    Создает шаблон пользовательского промпта на основе данных из prompts.yaml.
    
    Args:
        prompt_version (str): Версия промпта для использования. По умолчанию "current".
                             Если "current", используется версия из поля current.
    
    Returns:
        HumanMessagePromptTemplate: Объект HumanMessagePromptTemplate из LangChain
    """
    # Загружаем данные промптов
    prompts_data = load_prompts()
    style_guide = load_style_guide()
    format_fields = style_guide.get("format", {}).get("fields", {})
    format_fields = json.dumps(format_fields, ensure_ascii=False, indent=4)
    
    # Определяем версию промпта
    if prompt_version == "current":
        version = prompts_data["prompts"]["current"]
    else:
        version = prompt_version
    
    # Получаем данные для выбранной версии
    if version not in prompts_data["prompts"]:
        raise ValueError(f"Версия промпта '{version}' не найдена в файле prompts.yaml")
    
    prompt_config = prompts_data["prompts"][version]
    
    # Проверяем наличие пользовательского промпта
    if "user" not in prompt_config:
        raise ValueError(f"Пользовательский промпт не найден для версии '{version}'")
    
    # Создаем и возвращаем шаблон пользовательского сообщения
    return HumanMessagePromptTemplate.from_template(prompt_config["user"], partial_variables=format_fields)


def create_chat_prompt_template(prompt_version: str = "current") -> ChatPromptTemplate:
    """
    Создает объект ChatPromptTemplate на основе данных из prompts.yaml.
    
    Args:
        prompt_version (str): Версия промпта для использования. По умолчанию "current".
                             Если "current", используется версия из поля current.
    
    Returns:
        ChatPromptTemplate: Объект ChatPromptTemplate из LangChain
    """
    # Создаем список сообщений для ChatPromptTemplate
    messages = []
    
    # Добавляем системное сообщение, если оно есть
    try:
        system_template = create_system_prompt_template(prompt_version)
        messages.append(system_template)
    except ValueError:
        # Системный промпт не найден, пропускаем
        pass
    
    # Добавляем пользовательское сообщение, если оно есть
    try:
        user_template = create_user_prompt_template(prompt_version)
        messages.append(user_template)
    except ValueError:
        # Пользовательский промпт не найден, пропускаем
        pass
    
    # Проверяем, что есть хотя бы одно сообщение
    if not messages:
        raise ValueError(f"Не найдено ни одного промпта для версии '{prompt_version}'")
    
    # Создаем и возвращаем ChatPromptTemplate
    return ChatPromptTemplate.from_messages(messages)

class ChatBot:
    """
    Класс для работы с чат-ботом, использующий промпты из brand_chain.py
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2, request_timeout: int = 15):
        """
        Инициализация чат-бота
        
        Args:
            model_name (str): Название модели OpenAI
            temperature (float): Температура для генерации
            request_timeout (int): Таймаут запроса
        """
        self.session_id = str(uuid.uuid4())
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.request_timeout = request_timeout
        
        # Загружаем данные
        self.style_guide = load_style_guide()
        self.faq_data = self._load_faq_data()
        self.orders_data = self._load_orders_data()
        
        # Создаем модель и цепочку с памятью
        self.llm = ChatOpenAI(
            model_name=self.model_name, 
            temperature=self.temperature, 
            request_timeout=self.request_timeout
        )
        self.memory = ConversationBufferMemory(output_key="response")
        self.conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory, 
            return_final_only=False
        )
        
        # Добавляем системное сообщение
        self._setup_system_message()
        
        logging.info(f"=== New session {self.session_id} ===")
    
    def _load_faq_data(self) -> dict:
        """Загружает данные FAQ из файла"""
        faq_path = Path(__file__).parent.parent / "data" / "faq.json"
        if faq_path.exists():
            with open(faq_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_orders_data(self) -> dict:
        """Загружает данные заказов из файла"""
        orders_path = Path(__file__).parent.parent / "data" / "orders.json"
        if orders_path.exists():
            with open(orders_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _setup_system_message(self):
        """Настраивает системное сообщение для чат-бота"""
        # Создаем предзаполненный системный промпт
        system_message = create_system_prompt_template()
        
       
        self.conversation.memory.chat_memory.add_message(system_message.format())
    
    def get_order_status(self, order_id: str) -> str:
        """Получает статус заказа по ID из orders.json"""
        order_info = self.orders_data.get(order_id, f"Заказ с номером {order_id} не найден. Пожалуйста, проверьте правильность номера заказа.")
        return order_info
    
    def save_session(self, user_input: str, bot_reply: str, total_tokens: int = 0):
        """Сохраняет сессию в файл"""
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        with open(logs_dir / f"session_{self.session_id}.jsonl", "a", encoding='utf-8') as f:
            json.dump({
                "dialog": f"User: {user_input}\nBot: {bot_reply}", 
                "usage": total_tokens
            }, f, ensure_ascii=False)
            f.write("\n")
    
    def chat(self, user_input: str) -> str:
        """
        Обрабатывает пользовательский ввод и возвращает ответ бота
        Использует пайплайн: user_prompt | llm
        
        Args:
            user_input (str): Ввод пользователя
            
        Returns:
            str: Ответ бота
        """
        # Проверяем команду /order
        if user_input.startswith("/order "):
            order_id = user_input.replace("/order ", "").strip()
            return self.get_order_status(order_id)
        
        # Создаем пайплайн: user_prompt | llm
        # 1. Создаем шаблон пользовательского промпта
        user_prompt_template = create_user_prompt_template()
      
        chain = user_prompt_template | self.llm
        # 2. Вызываем LLM с отформатированным промптом
        response = self.llm.invoke({"user_input": user_input})
        
        # 3. Возвращаем содержимое ответа
        return response.content
    


BASE = Path(__file__).parent.parent
STYLE = load_style_guide()

