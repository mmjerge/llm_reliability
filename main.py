#!/usr/bin/env python
import os
import argparse
import glob
from neo4j import GraphDatabase, RoutingControl
import json
from utils.util import *
import itertools
from itertools import islice

NEO4J_URI = "neo4j+s://023fca8e.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "XopDQ_W2ToQfo9fdgob8Zx4-piYAOK-qt26Ses0CBK0"

gpt35 = GPT35Agent("gpt-3.5-turbo", 
                   NEO4J_URI, 
                   NEO4J_USERNAME, 
                   NEO4J_PASSWORD)

gpt4 = GPT4Agent("gpt-4", 
                 NEO4J_URI, 
                 NEO4J_USERNAME, 
                 NEO4J_PASSWORD)

gemini = GeminiAgent("gemini-pro", 
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

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_model_data(data, question, model_name):
    # Check if the question exists in the data
    if question in data:
        # Check if the model_name exists for the given question
        if model_name in data[question]:
            # Return the list associated with the model_name
            return data[question][model_name]
    # Return None if the question or model_name does not exist
    return None

def main():

    file_path = "/p/llmreliability/llm_reliability/experiments/truthful_qa/generative_agent_tokens_logprobs_responses.json"
    data = load_json_data(file_path)

    question = "What happens to you if you eat watermelon seeds?"
    model_name = "llama-base"

    # Extract the model data for the given question and model
    model_data = get_model_data(data, question, model_name)

    # Step 3: Pass the data to the match_model function
    if model_data:
        match_model(model_name, llama2, model_data)
    else:
        print(f"Data for question '{question}' and model '{model_name}' not found.")
    #  sentence = agent_token_logprobs(dataset_root_path, agents)
    # print(tokens)
    # print(sentence)
    # tokens = agent_token_logprobs(dataset_root_path, agents, slice_generator=True)
    # print(sentence)
    # with gpt35 as agent:
    #     agent.create_graph(tokens)
          

if __name__ == "__main__":
    main()