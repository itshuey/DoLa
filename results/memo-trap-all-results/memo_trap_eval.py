import json
import argparse

def import_jsonl_results(fp):
    with open(fp) as f:
        results = json.loads(f.readline())
    return results

def check_option(response, correct_option, strict=False):
    strict_correct = response == correct_option
    relax_correct = response.upper() == correct_option
    return strict_correct if strict else (strict_correct or relax_correct)

def get_correct_sentence(task):
    sentence_clause = task["prompt"].split(":", 1)[1]
    return f'{sentence_clause}{task["classes"][0]}'

def check_sentence(response, correct_answer, strict=False):
    strict_correct = response == correct_answer
    response = response.strip().lower()
    correct_answer = correct_answer.strip().lower()
    if response[-1] != '.' and correct_answer[-1] != '.':
        correct_answer = correct_answer[:-1]
    relax_correct = response == correct_answer

    return strict_correct if strict else (strict_correct or relax_correct)

def check_is_extended_option(response, options):
    return len(response) == 2 and response[0] in options and not response[1].isalpha()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="input.jsonl")
    args = parser.parse_args()
    file_path = args.data_path

    results = import_jsonl_results(file_path)
    num_binary, num_sentence, num_both = 0, 0, 0
    num_correct_binary, num_correct_sentence, num_correct_both = 0, 0, 0
    options = ['A', 'B']

    for i, response in enumerate(results['model_completion']):
        response = response.strip()
        if len(response) == 1 or check_is_extended_option(response, options):
            num_binary += 1
            # This will be either 'A' or 'B'
            correct_option = results['question'][i]['correct_option']
            num_correct_binary += check_option(response, correct_option)
        else:
            # Check for the case where both the option and word are returned
            response_split = response.split(' ', 1)
            potential_option = response_split[0]
            correct_sentence = get_correct_sentence(results['question'][i])
            if check_is_extended_option(potential_option, options):
                num_both += 1
                correct_option = results['question'][i]['correct_option']
                got_option_right = check_option(potential_option[0], correct_option)
                got_sentence_right = check_sentence(response_split[1], correct_sentence)
                num_correct_both += (got_option_right and got_sentence_right)
            else:
                num_sentence += 1
                num_correct_sentence += check_sentence(response, correct_sentence)


    num_total_correct = num_correct_binary + num_correct_sentence + num_correct_both
    num_total = len(results['model_completion'])
    print("Results for " + file_path)
    print(f'Accuracy: {num_total_correct/num_total:4f}, {num_total_correct}/{num_total}')
    print('\nBreakdown:')
    print(f'Options: {num_correct_binary}/{num_binary}')
    print(f'Sentences: {num_correct_sentence}/{num_sentence}')
    print(f'Both: {num_correct_both}/{num_both}')




