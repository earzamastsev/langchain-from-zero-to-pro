"""
Pydantic схемы для ответов чат-бота Shoply
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BotResponse(BaseModel):
    """
    Схема ответа чат-бота
    
    Содержит структурированный ответ бота с проверкой тона и рекомендуемыми действиями
    """
    answer: str = Field(
        ...,
        description="Краткий ответ на вопрос пользователя",
        min_length=1,
        max_length=500
    )
    
    tone: str = Field(
        ...,
        description="Контроль тона: совпадает ли тон (да/нет) + одна фраза почему",
        min_length=1,
        max_length=200
    )
    
    actions: List[str] = Field(
        ...,
        description="Список следующих шагов для клиента (0-3 пункта)",
        max_items=3
    )
