import os
import json
import numpy as np
from tqdm import tqdm
from BioASQ import DataLoader
import coco_evaluation2 as coco_eval


def read_dict(directory, filename):
    with open(os.path.join(directory, filename)) as handle:
        data = handle.read()
    dictionary = json.loads(data)

    return dictionary


def write_dict(dict, directory, filename):
    with open(os.path.join(directory, filename), 'w') as convert_file:
        convert_file.write(json.dumps(dict))


def serialize(list):
    list_contents = ""
    for list_item in list:
        list_contents += (list_item + " ")

    return list_contents.strip()


keys = {'BioGPT_Large_checkpoint.txt': 'BioGPT_answer', 'BioGPT_Large_HF_checkpoint.txt': 'BioGPT_HF_answer',
        'BioGPT_Large_HF_PubMed_checkpoint.txt': 'BioGPT_HF_QA_answer', 'ground_truth': 'ideal_answer'}

def extract_answers(dataset, split):
    answers = {}
    for question in dataset:
        answers[question['id']] = serialize(question[keys[split]]) if split == "ground_truth" \
                                  else question[keys[split]]

    return answers


def extract_snippets(gt_data):
    answers = {}
    for question in gt_data:
        rel_snippets = [snippet['text'] for snippet in question['snippets']]
        answers[question['id']] = serialize(rel_snippets)

    return answers


def extract_best_snippets(gt_data, lm_answers):
    answers = {}
    for question in gt_data:
        rel_snippets = [snippet['text'] for snippet in question['snippets']]
        snippets_relevance = [coco_eval.compute_scores({'caption': snippet}, {'caption': lm_answers[question['id']]},
                                                       logging=False)["ROUGE_L"] for snippet in rel_snippets]
        answers[question['id']] = rel_snippets[np.argmax(np.array(snippets_relevance))]

    return answers


def coco_eval_trainset():
    # Set the corresponding directories
    split = "train"
    predictions_path = "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL SYSTEMS/BioASQ11/LargeLanguageModels"
    val_set_path = "/media/georg_mosh/Data SSD/AUEB PROOFTREE GENERATION/NLProofS/data/entailment_trees_emnlp2021_data_v3/" \
                   "dataset/task_2/dev.jsonl"

    # Run the approximate evaluation
    loader = DataLoader()
    data = loader.data['questions']
    ground_truth = extract_answers(data, "ground_truth")
    experiments_names = os.listdir(predictions_path)
    for experiment in tqdm(experiments_names):
        exp_responses = read_dict(predictions_path, experiment)
        exp_results = extract_answers(exp_responses, experiment)
        print("\nExperiment: " + experiment + "\n")
        coco_eval.compute_scores(ground_truth, exp_results)


def coco_eval_testset(method="few_shot_prompting", split="BioASQ-task11bPhaseB-testset4.json",
                      runID="BioGPT_Large_HF_PubMed_test4.txt"):
    # Load the Large Language Model's respective generated answers
    exp_results = read_dict(os.path.join("/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results", method), runID)

    # Assume the relevant snippets as ground truth captions
    loader = DataLoader(split=split)
    data = loader.data['questions']
    ground_truth = extract_snippets(data)

    # Compute the evaluation metrics
    coco_eval.compute_scores(ground_truth, exp_results)


def coco_eval_selective_testset(method="few_shot_prompting", split="BioASQ-task11bPhaseB-testset4.json",
                                runID="BioGPT_Large_HF_PubMed_test4.txt"):
    # Load the Large Language Model's respective generated answers
    exp_results = read_dict(os.path.join("/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results", method), runID)

    # Assume the relevant snippets as ground truth captions
    loader = DataLoader(split=split)
    data = loader.data['questions']
    ground_truth = extract_best_snippets(data, exp_results)

    # Compute the evaluation metrics
    coco_eval.compute_scores(ground_truth, exp_results)

if __name__ == "__main__":
    coco_eval_trainset()
    # coco_eval_testset()
    # coco_eval_selective_testset()
