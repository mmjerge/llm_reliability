#!/usr/bin/env python
import os
import argparse
import glob
from neo4j import GraphDatabase, RoutingControl
import json
from utils.util import *
import itertools
from itertools import islice
from tqdm import tqdm

NEO4J_URI = "neo4j+s://023fca8e.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "null"

gpt35 = GPT35Agent("gpt-3.5-turbo-0125", 
                   NEO4J_URI, 
                   NEO4J_USERNAME, 
                   NEO4J_PASSWORD)

gpt4 = GPT4Agent("gpt-4", 
                 NEO4J_URI, 
                 NEO4J_USERNAME, 
                 NEO4J_PASSWORD)

llama2 = LlamaBaseAgent("llama-base", 
                        neo4j_uri=NEO4J_URI, 
                        neo4j_username=NEO4J_USERNAME, 
                        neo4j_password=NEO4J_PASSWORD)

llama2large = LlamaLargeAgent("llama-large",
                              neo4j_uri=NEO4J_URI, 
                              neo4j_username=NEO4J_USERNAME, 
                              neo4j_password=NEO4J_PASSWORD)

mistral = MistralAgent("mistral-base",
                       neo4j_uri=NEO4J_URI, 
                       neo4j_username=NEO4J_USERNAME, 
                       neo4j_password=NEO4J_PASSWORD)

phi2 = MicrosoftAgent("phi-2",
                    neo4j_uri=NEO4J_URI, 
                    neo4j_username=NEO4J_USERNAME, 
                    neo4j_password=NEO4J_PASSWORD)

agents = [llama2]

llama_model_path = "/p/llmreliability/llm_reliability/models/meta/base/meta-llama/Llama-2-7b-hf"

llama_large_model_path = "/p/llmreliability/llm_reliability/models/meta/large/meta-llama/Llama-2-70b-hf"

mistral_model_path = "/p/llmreliability/llm_reliability/models/mistral/base/mistralai/Mistral-7B-v0.1"

phi2_model_path = "/p/llmreliability/llm_reliability/models/microsoft/base/microsoft/phi-2"

questions = read_bill("/p/llmreliability/llm_reliability/data/benchmarks/billsum/data/ca_test-00000-of-00001.parquet")

bill_dict = {}
for bill in tqdm(islice(questions, 10), desc='Processing articles.'):
    responses = {}
    for agent in tqdm(agents, desc='Prompting agents.', leave=False):
        agent_name = agent.model_name
        if agent_name == "gpt-3.5-turbo-0125" or agent_name == "gpt-4":
            intialize = agent.start_chat(f"Please summarize this bill: {bill}")
            responses[agent_name] = agent.give_response(intialize)
        elif agent_name == "llama-base":
            responses[agent_name] = agent.generate_text(f"Please summarize the following article. {bill}", 
                                                        llama_model_path) 
                                                        # return_raw_outputs=True)
        elif agent_name == "llama-large":
            responses[agent_name] = agent.generate_text(f"Please summarize this bill: {bill}", 
                                                        llama_large_model_path,
                                                        return_raw_outputs=True)
        elif agent_name == "mistral-base":
            responses[agent_name] = agent.generate_text(f"Please summarize this bill: {bill}", 
                                                        mistral_model_path)
        elif agent_name == "phi-2":
            responses[agent_name] = agent.generate_text(f"Please summarize the following article. {article}", 
                                                        phi2_model_path)
        else:
            pass
    bill_dict[bill] = responses
print("Summarization complete.")

file_path = "llama2_summarization_responses.json"

with open(file_path, 'w') as file:
    json.dump(bill_dict, file, indent=4)
