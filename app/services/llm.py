import os
from openai import AsyncOpenAI
import asyncio
from ..core.celery_app import celery_app
from ..configs import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE



async def get_llm_response_async(messages: list, lang: str = "ru") -> str:
    """Асинхронный вызов LLM DeepSeek chat completions"""
    
    client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
    )

    model = "deepseek-chat"
    if lang == "ru":
        system_prompt = (
            "Ты ассистент, который кратко отвечает на вопросы и помогает вести диалог. "
            "Если пользователь делает ошибки, исправь их и напиши правильный вариант. "
            "Отвечай в формате: Ответ: ... Исправление: ... "
            "Если ошибок нет, то просто отвечай без совета"
        )
    else:
        system_prompt = (
            "You are an assistant who briefly answers questions and helps keep the conversation. "
            "If the user makes mistakes, correct them and write the correct version. "
            "Always reply in the format: Answer: ... Correction: ... "
            "If there are no mistakes, just repeat the user's phrase in the correction."
        )
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = await client.chat.completions.create(
        model=model,
        messages=full_messages
    )
    return response.choices[0].message.content

@celery_app.task(ignore_result=False)
def get_llm_response_task(messages: list, lang: str = "ru") -> str:
    """Celery task: вызывает DeepSeek LLM и возвращает ответ."""
    return asyncio.run(get_llm_response_async(messages, lang))
