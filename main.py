#!/usr/bin/env python
import os
import argparse
from neo4j import GraphDatabase, RoutingControl
import json
from utils.util import *

NEO4J_URI = "neo4j+s://023fca8e.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "XopDQ_W2ToQfo9fdgob8Zx4-piYAOK-qt26Ses0CBK0"

gpt35 = GPT35Agent("gpt-3.5-turbo", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
gpt4 = GPT4Agent("gpt-4", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
# palm = GeminiAgent("text-bison-001", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
agents = [gpt35, gpt4]

questions = read_questions("/p/llmreliability/llm_reliability/data/truthful_qa/generation/validation-00000-of-00001.parquet")

def main():
    qa_data = {}
    for question in questions:
        responses = {}
        for agent in agents:
            agent_name = agent.model_name
            try:
                initialize = agent.start_chat(question)
                responses[agent_name] = agent.give_response(initialize)
            except Exception as e:
                print(f"Error with agent {agent_name}: {e}")
        qa_data[question] = responses
        for agent_name, response in responses.items():
            print(f"{agent_name}: {response}")
    print("QA complete.")

    try:
        file_path = 'agent_responses.json'
        with open(file_path, 'w') as file:
            json.dump(qa_data, file, indent=4)
        print(f"File written to {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"Error writing file: {e}")
          

if __name__ == "__main__":
    main()