from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import os
import json
import numpy as np

def evaluate_results(answer_string, question, config):
    """
    Evaluates the generated answer and logs the results to the prediction table.

    Args:
        answer_string (str): The generated answer string.
        question (dict): The question data.
        config (dict): The configuration dictionary.
        prediction_table (wandb.Table): The wandb prediction table.
        userprompt (str): The user prompt string.

    Returns:
        bleu (int): The BLEU score.
        rouge (int): The ROUGE score.
        correct (bool): The number of correct answers.
    """

    if config['data']['data_id'] == 'NQ':
        gold_answers = question['answers']
    elif config['data']['data_id'] == 'QuALITY':
        gold_answers = str(question['questions'][0]['gold_label'])

    bleu = calculate_bleu(gold_answers, answer_string)

    if isinstance(gold_answers, str):
        rouge = calculate_rouge(gold_answers, answer_string)['rougeL'].fmeasure
    else:
        rouge = np.mean([calculate_rouge(g, answer_string)['rougeL'].fmeasure for g in gold_answers])
    
    correct += evaluate_answer(answer_string, gold_answers)

    # Add data to the prediction table
    return bleu, rouge, correct, gold_answers


def evaluate_answer(answer_string, gold_answers):
    if isinstance(gold_answers, str):
        return int(answer_string == gold_answers)
    elif isinstance(gold_answers, list):
        return int(answer_string in gold_answers)
    else:
        raise ValueError("Invalid gold answer format")

def save_results(results, outputs_path, dataset_id):
    with open(os.path.join(outputs_path, f'results_{dataset_id}.json'), 'w') as f:
        json.dump(results, f)

def calculate_bleu(reference, candidate):
    if isinstance(reference, str):
        reference = [reference]
    # Smoothing function for short answers
    smoothie = SmoothingFunction().method4

    # Split the sentences into words
    reference = [r.split() for r in reference]
    candidate = candidate.split()

    # Calculate the BLEU score with smoothing
    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    return score

def calculate_rouge(reference, candidate):
    # Initialize a ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Calculate the ROUGE-L score
    scores = scorer.score(reference, candidate)

    return scores

