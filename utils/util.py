import os
import json
import pandas as pd
import numpy as np
import pandas as pd
import torch
import neo4j
import spacy
from typing import Dict, Union
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseAgent:
    """Base class for creating different llm baesd agents
    """
    def __init__(self, name: str, prompt: str, token=None) -> None:
        self.name = name
        self.prompt = prompt

class GPT35Agent(BaseAgent):
    """GPT 3.5 Turbo Agent Class

    Parameters
    ----------
    BaseAgent : class
        Super class for all agents
    """
    def __init__(self, name: str, prompt: str, token=None) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            _description_
        prompt : str
            _description_
        token : _type_, optional
            _description_, by default None
        """
        super().__init__(name, prompt, token)
        self.client = OpenAI(api_key=token)
        self.model = "gpt-3.5-turbo"

    def start_chat(self, message: str):
        """_summary_

        Parameters
        ----------
        message : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        try:
            chat = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model=self.model,
                logprobs=True, 
                max_tokens=200,
                temperature=.5
            )
            return chat
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")

    def give_response(self, chat) -> str:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        str
            _description_
        """
        content = chat.choices[0].message.content
        return content
    
    def retrieve_llm_tokens(self, chat) -> dict[int, str]:
        """_summary_

        Parameters
        ----------
        chat : _type_
            _description_

        Returns
        -------
        dict[int, str]
            _description_
        """
        token_dict = {count:value.token for count, value in enumerate(chat.choices[0].logprobs.content)} 
        return token_dict
    
    def get_nlp_properties(self, content) -> Dict[int, str, Union[str, str]]:
        """_summary_

        Parameters
        ----------
        content : _type_
            _description_

        Returns
        -------
        Dict[int, str, Union[str, str]]
            _description_
        """
        nlp = spacy.load("en_core_web_sm")
        document = nlp(content)
        property_dict = {}
        for index, token in enumerate(document):
            property_dict[index] = {'text': token.text, 
                                    'lemma': token.lemma_, 
                                    'pos': token.pos_, 
                                    'tag': token.tag_, 
                                    'dep': token.dep_, 
                                    'shape': token.shape_, 
                                    'is_alpha': token.is_alpha, 
                                    'is_stop': token.is_stop}
        return property_dict
    
    def create_token_node(self, tx, token):
        """_summary_

        Parameters
        ----------
        tx : _type_
            _description_
        token : _type_
            _description_
        """
        tx.run("MERGE (t:Token {name: $token})", token=token)

    def create_weighted_relationship(self, tx, token1, token2, weight):
        """_summary_

        Parameters
        ----------
        tx : _type_
            _description_
        token : _type_
            _description_
        """
        tx.run("""
            MATCH (t1:Token {name: $token1}), (t2:Token {name: $token2})
            MERGE (t1)-[r:NEXT]->(t2)
            ON CREATE SET r.weight = $weight
            ON MATCH SET r.weight = $weight
        """, token1=token1, token2=token2, weight=weight)
                

class GPT4Agent(BaseAgent):
    def __init__(self, name: str, prompt: str, token=None) -> None:
        super().__init__(name, prompt, token)

    def start_chat(self, message: str):
        try:
            chat = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model=self.model,
                logprobs=True, 
                max_tokens=200,
                temperature=.5
            )
            return chat
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")


class LlamaBaseAgent(BaseAgent):
    def __init__(self, name: str, prompt: str, token=None) -> None:
        super().__init__(name, prompt, token)

class LlamaLargeAgent(BaseAgent):
    def __init__(self, name: str, prompt: str, token=None) -> None:
        super().__init__(name, prompt, token)

class MicrosoftAgent(BaseAgent):
    def __init__(self, name: str, prompt: str, token=None) -> None:
        super().__init__(name, prompt, token)

class MistralAgent(BaseAgent):
    def __init__(self, name: str, prompt: str, token=None) -> None:
        super().__init__(name, prompt, token)
        
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


