from neo4j import GraphDatabase
import json
from itertools import islice
from tqdm import tqdm

def add_token_pair(tx, index, index2, question, model_name, first_item_key, first_item_logprob, second_item_key, second_item_logprob):
    query = (
        "MERGE (t1:Token {name: $first_item_key, index: $index, question: $question}) "
        "MERGE (t2:Token {name: $second_item_key, index: $index2, question: $question}) "
        "MERGE (t1)-[r:NEXT_TOKEN {model: $model_name, logprob: $first_item_logprob}]->(t2)"
    )
    tx.run(query, index=index, index2=index2, question=question, model_name=model_name,
           first_item_key=first_item_key, first_item_logprob=first_item_logprob,
           second_item_key=second_item_key, second_item_logprob=second_item_logprob)

def inspect_json(json_data, driver):
    with driver.session() as session:
        for question, models in tqdm(islice(json_data.items(), 10)):
            for model_name, tokens in tqdm(models.items()):
                for i in range(len(tokens) - 1):
                    first_item = tokens[i]
                    second_item = tokens[i + 1]
                    for (key1, value1), (key2, value2) in zip(first_item.items(), second_item.items()):
                        if isinstance(value1, dict) and isinstance(value2, dict):
                            for k1, v1 in value1.items():
                                k1_token_name = k1
                                v1_token_logprob = v1
                            for k2, v2 in value2.items():
                                k2_token_name = k2
                                v2_token_logprob = v2
                            session.execute_write(add_token_pair, 
                                                  key1,
                                                  key2, 
                                                  question, 
                                                  model_name, 
                                                  k1_token_name, 
                                                  v1_token_logprob, 
                                                  k2_token_name, 
                                                  v2_token_logprob)            

def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "Xavier21!"

    json_file_path = '/Users/michaeljerge/Documents/GitHub/llm_reliability/experiments/truthful_qa/generative_agent_tokens_logprobs_responses.json'
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        inspect_json(json_data, driver)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
