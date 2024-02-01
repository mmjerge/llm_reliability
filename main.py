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

gpt35 = GPT35Agent("gpt-3.5-turbo", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
gpt4 = GPT4Agent("gpt-4", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
gemini = GeminiAgent("gemini-pro", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
agents = [gpt35, gpt4]

dataset_root_path = "/p/llmreliability/llm_reliability/data/"

def agent_truthfulqa_responses(dataset_root_path: str, agent_list: list, slice_generator=False):
    files = glob.glob(dataset_root_path + "/**/*.parquet", recursive=True)
    questions = itertools.chain.from_iterable(read_questions(file) for file in files)
    if slice_generator:
        truthfulqa_responses = {}
        for question in islice(questions, 1):
            responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        responses[agent_name] = agent.give_response(initialize)
                    else:
                        responses[agent_name] = agent.start_chat(question)
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_responses[question] = responses
        print("QA complete.")
        return truthfulqa_responses
    else:
        truthfulqa_responses = {}
        for question in questions:
            responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        responses[agent_name] = agent.give_response(initialize)
                    else:
                        responses[agent_name] = agent.start_chat(question)
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_responses[question] = responses
        print("QA complete.")
        return truthfulqa_responses

def agent_token_logprobs(dataset_root_path: str, agent_list: list, slice_generator=False):
    files = glob.glob(dataset_root_path + "/**/*.parquet", recursive=True)
    questions = itertools.chain.from_iterable(read_questions(file) for file in files)
    if slice_generator:
        truthfulqa_tokens_dict = {}
        for question in islice(questions, 1):
            token_responses = {}
            for agent in agent_list:
                agent_name = agent.model_name
                try:
                    if agent_name == "gpt-3.5-turbo" or agent_name == "gpt-4":
                        initialize = agent.start_chat(question)
                        token_responses[agent_name] = agent.retrieve_tokens_logprobs(initialize)
                    else:
                        pass
                except Exception as e:
                    print(f"Error with agent {agent_name}: {e}.")
            truthfulqa_tokens_dict[question] = token_responses
        print("QA token generation complete.")
        return truthfulqa_tokens_dict
    else: 
        pass

def main():
    sentence = agent_truthfulqa_responses(dataset_root_path, agents, slice_generator=True)
    tokens = agent_token_logprobs(dataset_root_path, agents, slice_generator=True)
    print(sentence)
    with gpt35 as agent:
        agent.create_graph(tokens)

    # try:
    #     file_path = 'gpt_tokens_logprobs_responses.json'
    #     with open(file_path, 'w') as file:
    #         json.dump(tokens, file, indent=4)
    #     print(f"File written to {os.path.abspath(file_path)}")
    # except Exception as e:
    #     print(f"Error writing file: {e}")
          

if __name__ == "__main__":
    main()