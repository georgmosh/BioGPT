import os
import coco_evaluation2 as coco_eval
from BioASQ import DataLoader
from evaluation import extract_snippets
from helpers import *

import numpy as np

def coco_eval_testset(method="few_shot_prompting", split="BioASQ-task11bPhaseB-testset4.json",
                      runID="BioGPT_Large_HF_PubMed_test4.txt", preprocess=False):
    # Load the Large Language Model's respective generated answers
    exp_results = read_dict(os.path.join(r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results", method), runID)

    # Assume the relevant snippets as ground truth captions
    loader = DataLoader(split=split)
    data = loader.data['questions']
    ground_truth = extract_snippets(data)

    # Apply preprocessing to the relevant snippets if prompted.
    if preprocess:
        for q_id in ground_truth.keys():
            ground_truth[q_id] = re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', ground_truth[q_id].lower())

    # Compute the evaluation metrics
    exp_metrics = coco_eval.compute_scores(ground_truth, exp_results)

    return exp_metrics

data = ["BioASQ-task11bPhaseB-testset1.json", "BioASQ-task11bPhaseB-testset2.json",
        "BioASQ-task11bPhaseB-testset3.json", "BioASQ-task11bPhaseB-testset4.json"]

zsl_scores = {}
zsl_rouge_recall = {}
zsl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting_processed"
for results_file in os.listdir(zsl_dir):
    metrics = coco_eval_testset(method="zero_shot_prompting", split=data[int(results_file.split("test")[1].split("_")[0])-1],
                                runID=results_file, preprocess=True)
    zsl_scores[results_file] = metrics
    zsl_rouge_recall[results_file] = np.average(np.array(list(metrics.values())))

zsl_rouge_recall = dict(sorted(zsl_rouge_recall.items(), key=lambda item: item[1], reverse=True))

icl_scores = {}
icl_rouge_recall = {}
icl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\few_shot_prompting_processed"
for results_file in os.listdir(icl_dir):
    metrics = coco_eval_testset(method="few_shot_prompting", split=data[int(results_file.split("test")[1].split("_")[0])-1],
                                runID=results_file, preprocess=True)
    icl_scores[results_file] = metrics
    icl_rouge_recall[results_file] = np.average(np.array(list(metrics.values())))

icl_rouge_recall = dict(sorted(icl_rouge_recall.items(), key=lambda item: item[1], reverse=True))

print("Metrics are successfully computed!")
