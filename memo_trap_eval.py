# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import argparse
import ast
import json
import os
import re
from tqdm import tqdm

import pandas as pd
import transformers
from transformers import logging as transformers_logging

from dola_t5 import DoLa

# Set Transformers logging to error only to reduce noise
transformers_logging.set_verbosity_error()

# Constants
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
N_SHOT = 7
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"


def parse_classes(classes_str):
    """Parse a string representation of classes into a list."""
    classes_str = classes_str.strip('"[]')
    classes_list = classes_str.split("', '")
    classes_list = [item.strip().strip("'") for item in classes_list]
    return classes_list


def load_csv(file_path):
    """Load data from a CSV file."""
    list_data = []
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            first_comma = line.find(",")
            last_comma = line.rfind(",")
            prompt = line[:first_comma].strip('"')
            classes_str = line[first_comma + 1 : last_comma].strip('"')
            answer_index = int(line[last_comma + 1 :])
            classes = parse_classes(classes_str)
            data = {"prompt": prompt, "classes": classes, "answer_index": answer_index}
            list_data.append(data)
    return list_data


def load_jsonl(file_path, only_part_1=True):
    """Load data from a JSONL file."""
    data_list = []
    options = ["A", "B"]
    with open(file_path, "r") as f:
        for line in f:
            cur_data_json = json.loads(line)
            cur_data_json["classes"] = ast.literal_eval(cur_data_json["classes"])
            cur_data_json["correct_option"] = options[cur_data_json["answer_index"]]
            cur_data_json["correct_ending"] = cur_data_json["classes"][
                cur_data_json["answer_index"]
            ]
            if cur_data_json["part"] == 1 or not only_part_1:
                data_list.append(cur_data_json)
    return data_list


def extract_and_compare_answer(question, model_completion):
    """Extract and compare the model's answer with the correct answer."""
    prompt_ending = question["prompt"].split()[-1]
    model_answer_ending = model_completion.split(prompt_ending)[-1].strip()
    correct_answer = question["classes"][question["answer_index"]].strip()
    is_correct = model_answer_ending == correct_answer
    return is_correct, model_answer_ending, correct_answer


def create_demo_text():
    """Create demo text for the prompt."""
    questions_and_answers = [
        (
            'Write a quote that ends in the word "discovers": He who searches,',
            "He who searches, discovers.",
        ),
        (
            'Write a quote that ends in the word "outlook": To bad times, a good',
            "To bad times, a good outlook.",
        ),
        (
            'Write a quote that ends in the word "opening": Flies don\'t enter a closed',
            "Flies don't enter a closed opening.",
        ),
    ]
    demo_text = "Consider the ending word of each quote and complete it, pay attention to the instructions you are being asked to follow.\n\n"
    for question, answer in questions_and_answers:
        demo_text += f"Q: {question}\nA: {answer}\n\n"
    return demo_text


def build_prompt(sample, version=3):
    """Build prompt based on the sample and version."""
    if version == 3:
        prompt = f'For the question: {sample["prompt"]}...\n'
        sentence_clause = sample["prompt"].split(":", 1)[1]
        prompt += "Which is the correct answer?"
        prompt += f' A){sentence_clause}{sample["classes"][0]}'
        prompt += f' B){sentence_clause}{sample["classes"][1]}'
    else:
        raise ValueError("Unsupported prompt version")
    return prompt

# new code that needs updating
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max-gpu-memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--relative-top", type=float, default=0.1)
    parser.add_argument("--print-logits", action="store_true")
    return parser.parse_args()

def main(config):
    """Main function to execute the script logic."""
    # Load data
    try:
        list_data_dict = load_jsonl(config.data_path, only_part_1=True)  # Load only part 1 data if applicable
    except FileNotFoundError:
        print(f"Error: File {config.data_path} not found.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    # Debug mode: process a smaller subset of data
    if config.debug:
        list_data_dict = list_data_dict[:15]

    # Handle parallel execution
    if config.parallel and config.shard_id is not None:
        chunk_size = len(list_data_dict) // config.total_shard
        list_data_dict = list_data_dict[config.shard_id * chunk_size: (config.shard_id + 1) * chunk_size]

    # Initialize model
    try:
        llm = DoLa(config.model_name, config.device, config.num_gpus, config.max_gpu_memory)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    result_dict = {
        "question": [],
        "model_completion": [],
        "input_text": [],
        "model_answer_ending": [],
        "correct_answer": [],
        "correctness": [],
    }

    # Generate responses and evaluate
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample)
        try:
            model_completion = llm.generate(input_text, max_new_tokens=config.max_new_tokens, top_p=config.top_p,
                                            top_k=config.top_k, temperature=config.temperature,
                                            repetition_penalty=config.repetition_penalty)
        except Exception as e:
            print(f"Error generating model completion: {e}")
            continue

        is_correct, model_answer_ending, correct_answer = extract_and_compare_answer(sample, model_completion)

        # Append results
        result_dict["question"].append(sample)
        result_dict["model_completion"].append(model_completion)
        result_dict["input_text"].append(input_text)
        result_dict["model_answer_ending"].append(model_answer_ending)
        result_dict["correct_answer"].append(correct_answer)
        result_dict["correctness"].append(is_correct)

        if config.debug:
            print(f"Question: {input_text}")
            print(f"Model Completion: {model_completion}")
            print(f'Correct Option: {sample["correct_option"]},{sample["correct_ending"]}')

    # Save results
    output_file = config.output_path if config.shard_id is None else f"{config.output_path}_{config.shard_id}.jsonl"
    with open(output_file, "w") as f:
        json.dump(result_dict, f)

    if config.debug:
        print("Processing complete. Results saved.")

if __name__ == "__main__":
    config = parse_arguments()
    main(config)