import os
from abc import ABCMeta, abstractmethod
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


#TODO Make Abstract class and have base agent inherit from that
#TODO Add additional classes for Claude and Google LLMs

class BaseAgent():
    """Base class for creating different llm baesd agents
    """
    def __init__(self, 
                 model_name: str,
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str) -> None:
        self.model_name = model_name
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

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
        response = self.API_RESPONSE_ERROR
        for _ in range(self.API_MAX_RETRY):
            try:
                chat = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": message
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
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.client = OpenAI(api_key=api_key)

    def start_chat(self, message: str):
        try:
            chat = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                model=self.model_name,
                logprobs=True, 
                max_tokens=200,
                temperature=.5
            )
            return chat
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")


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
                 cuda_device: str="cuda",
                 cache_path: Optional[str]=None) -> None:
        super().__init__(model_name, neo4j_uri, neo4j_username, neo4j_password)
        self.cuda_device = cuda_device
        self.cache_path = cache_path

    def get_devices(self):
        cuda = torch.cuda.is_available()
        device = (lambda: torch.cuda.device(self.cuda_device) if cuda else torch.device("cpu"))()
        num_gpus = (lambda: torch.cuda.device_count() if cuda else 0)()
        return device, num_gpus

    def tokenize(self, get_devices):
        devices = get_devices()
        model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     cache_dir=self.cache_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if devices[1] > 1:
            torch.distributed.init_process_group(backend='nccl')
            device = torch.device("cuda", rank)




class MistralAgent(BaseAgent):
    def __init__(self, model_name: str, prompt: str, token=None) -> None:
        super().__init__(model_name, prompt, token)

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


