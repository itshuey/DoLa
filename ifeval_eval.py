import argparse
from tqdm import tqdm

from transformers import logging as transformers_logging
from dola_t5 import DoLa

def load_jsonl(file_path):
    """ Load data from a JSONL file """
    with open(file_path, 'r') as f:
        return [json.loads(line)['prompt'] for line in f]

def create_demo_text():
    """ Create demonstration text for prompts """
    questions_answers = [
        ("Write a sentence describing the flavor of coffee. Make sure the word 'roasted' appears at least two times in the sentence, and include a bolded word. Like: *this is bolded text*.", 
         "The bold, *roasted* flavor of coffee envelopes the palate, infusing each sip with rich, *roasted* notes reminiscent of toasted caramel and dark chocolate."),
        ("List the months of the year using all capital letters.", 
         "JANUARY, FEBRUARY, MARCH, APRIL, MAY, JUNE, JULY, AUGUST, SEPTEMBER, OCTOBER, NOVEMBER, DECEMBER.")
    ]
    demo_text = 'Take note of the instructions and responses in the following examples:\n\n'
    for idx, (q, a) in enumerate(questions_answers):
        demo_text += f'Example {idx+1}:\nInstruction: {q}\nResponse: {a}\n\n'
    return demo_text

transformers_logging.set_verbosity(40)

def initialize_model(model_name, device, num_gpus, max_gpu_memory):
    """ Initialize the DoLa model with specified configurations """
    return DoLa(model_name, device, num_gpus, max_gpu_memory)

def generate_response(model, prompt, generate_kwargs):
    """ Generate model response for a given prompt """
    return model.generate(prompt, **generate_kwargs).strip()


def parse_arguments():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Run the DoLa model on prompts.')
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--output-path", type=str, default="./results")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model = initialize_model(args.model_name, args.device, args.num_gpus, args.max_gpu_memory)
    prompts = load_jsonl(args.data_path + '/ifeval-input-data.jsonl')
    
    results = []
    for prompt in tqdm(prompts):
        generate_kwargs = {
            'max_new_tokens': 1024,
            'top_p': 0.95,
            'top_k': 0,
            'temperature': 0.9,
            'repetition_penalty': 1.2,
            'print_logits': False
        }
        completion = generate_response(model, prompt, generate_kwargs)
        results.append({'prompt': prompt, 'response': completion})

        if args.debug:
            print(f'Question: {prompt}\nResponse: {completion}\n')

    with open(args.output_path + '/results.jsonl', 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')

if __name__ == "__main__":
    main()
