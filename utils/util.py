import os
from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import json
import time
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import neo4j
import spacy
from typing import Dict, Union, Optional
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai

class Connection(ABC):
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BaseAgent(Connection):
    """Base class for creating different llm baesd agents
    """
    def __init__(self,
                 model_name: str,
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.driver = None

    # To use with a "with" block
    def __enter__(self):
        try:
            auth = (self.neo4j_username, self.neo4j_password)
            self.driver = neo4j.GraphDatabase.driver(self.neo4j_uri, auth=auth)
            return self.driver
        except Exception as e:
            # If there is a connection error, this cleans up resources
            if self.driver is not None:
                self.driver.close()
            raise e
  
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver is not None:
            self.driver.close()

class GeminiAgent(BaseAgent):
    """_summary_

    Parameters
    ----------
    BaseAgent : _type_
        _description_
    """
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 1.0
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0
    API_KEY = os.getenv("GEMINI_API_KEY")
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 api_key=API_KEY) -> None:
        """_summary_

        Parameters
        ----------
        model_name : str
            _description_
        neo4j_uri : str
            _description_
        neo4j_username : str
            _description_
        neo4j_password : str
            _description_
        api_key : _type_, optional
            _description_, by default API_KEY
        """
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = genai.configure(api_key=api_key)
    
    def start_chat(self, prompt: str):
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                model = genai.GenerativeModel(
                    model_name = self.model_name
                )
                chat = model.start_chat()
                response = chat.send_message(prompt)
                break
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_RETRY_SLEEP)
        return response
    
    def count_tokens(self, response):
        content = response._raw_response.usage_metadata
        return content

class GPT35Agent(BaseAgent):
    #TODO add citation for adaptation of work in https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py

    """GPT 3.5 Turbo Agent Class

    Parameters
    ----------
    BaseAgent : class
        Super class for all agents
    """
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0
    API_KEY = os.getenv("OPENAI_API_KEY")
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 api_key=API_KEY) -> None:
        """_summary_

        Parameters
        ----------
        model_name : str
            _description_
        neo4j_uri : str
            _description_
        neo4j_username : str
            _description_
        neo4j_password : str
            _description_
        """
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = OpenAI(api_key=api_key)

    def start_chat(self, prompt: str):
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
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    logprobs=True, 
                    max_tokens=200,
                    temperature=.5, 
                    timeout=self.API_TIMEOUT
                )
                response = chat
                break
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_RETRY_SLEEP)
        return response

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
    
    def get_nlp_properties(self, content):
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
        tx.run("MERGE (t:Token {name: $token, model: $model})", 
               token=token, 
               model=self.model_name)

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
    #TODO add citation for adaptation of work in https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py

    """GPT 3.5 Turbo Agent Class

    Parameters
    ----------
    BaseAgent : class
        Super class for all agents
    """
    API_RETRY_SLEEP = 10
    API_RESPONSE_ERROR = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20.0
    API_KEY = os.getenv("OPENAI_API_KEY")
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 api_key=API_KEY) -> None:
        """_summary_

        Parameters
        ----------
        model_name : str
            _description_
        neo4j_uri : str
            _description_
        neo4j_username : str
            _description_
        neo4j_password : str
            _description_
        """
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = OpenAI(api_key=api_key)

    def start_chat(self, prompt: str):
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
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    logprobs=True, 
                    max_tokens=200,
                    temperature=.5, 
                    timeout=self.API_TIMEOUT
                )
                response = chat
                break
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_RETRY_SLEEP)
        return response

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


class LlamaBaseAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="cuda",
                 cache_path: Optional[str]=None) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

class LlamaLargeAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="cuda",
                 cache_path: Optional[str]=None) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

class MicrosoftAgent(BaseAgent):

    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str,
                 cuda_device: str="0",
                 cache_path: Optional[str]=None) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.device(f"cuda:{self.cuda_device}" if cuda else "cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus

    def tokenize(self, prompt: str):
        torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs)
        text = tokenizer.batch_decode(outputs)[0]
        return text
        # torch.set_default_device("cuda")
        # if self.get_devices()[1] >= 1:
        #     # torch.distributed.init_process_group(backend='nccl')
        #     # rank = distributed.get_rank()
        #     # world_size = distributed.get_world_size()
        #     # print(f"World size: {world_size}")
        #     device, _ = self.get_devices()
        #     torch.cuda.set_device(device)
        #     model.to(device)
        #     # model = DDP(model, device_ids=[rank])
        #     inputs = tokenizer(prompt, return_tensors="pt")
        #     input_ids = inputs['input_ids']
        #     with torch.no_grad():
        #         generated_outputs = model.generate(input_ids, max_length=max_length)
        #         generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        #     combined_text = prompt + generated_text
        #     combined_inputs = tokenizer.encode(combined_text, return_tensors="pt")
        #     with torch.no_grad():
        #         outputs = model(combined_inputs, labels=combined_inputs)
        #         logits = outputs.logits
        #     probabilities = torch.nn.functional.softmax(logits, dim=-1)
        #     log_probabilities = torch.log(probabilities)
        #     tokens = tokenizer.convert_ids_to_tokens(combined_inputs[0])
        #     log_probs_dict = {token: log_prob.item() for token, log_prob in zip(tokens, log_probabilities[0])}
        #     torch.distributed.destroy_process_group()
        #     return generated_text, log_probs_dict

class MistralAgent(BaseAgent):
    def __init__(self, 
                 model_name: str, 
                 neo4j_uri: str, 
                 neo4j_username: str, 
                 neo4j_password: str) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)

def read_questions(file, batch_size=1000):
    """_summary_

    Parameters
    ----------
    file : _type_
        _description_
    batch_size : int, optional
        _description_, by default 1000

    Yields
    ------
    _type_
        _description_
    """
    parquet_file = pq.ParquetFile(file)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['question']):
        table = batch.to_pandas()
        for question in table['question']:
            yield question
        
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


