import os
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model_name = model
        self.chat_model = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

