# rag_system/llms.py

from abc import ABC, abstractmethod
from langchain import OpenAI

class BaseLLM(ABC):
    @abstractmethod
    def get_llm(self):
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, temperature: float):
        self.api_key = api_key
        self.temperature = temperature

    def get_llm(self):
        return OpenAI(openai_api_key=self.api_key, temperature=self.temperature)
