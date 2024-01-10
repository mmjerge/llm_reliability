import os
import pandas as pd
import numpy as np
import pandas as pd
import os
from openai import OpenAI

class BaseAgent:
    """Base class for creating different llm baesd agents
    """
    def __init__(self, name, prompt, message, token=None) -> None:
        self.name = name
        self.prompt = prompt
        self.message = message

class GPT35Agent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)
        try:
            self.client = OpenAI(api_key=token)
            self.model = "gpt-3.5-turbo"
            self.chat = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model=self.model
            )
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")


class GPT4Agent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)
        try:
            self.client = OpenAI(api_key=token)
            self.model = "gpt-4"
            self.chat = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model=self.model
            )
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")

class LlamaBaseAgent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)

class LlamaLargeAgent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)

class MicrosoftAgent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)

class MistralAgent(BaseAgent):
    def __init__(self, name, prompt, message, token=None) -> None:
        super().__init__(name, prompt, message, token)

def match_model(model):
    match model:
        case "meta":
            pass
        case "mistral":
            pass
        case "microsoft":
            pass
        case _:
            pass


