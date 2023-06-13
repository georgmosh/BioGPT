import os

from helpers import *
from PubMed.BioASQ import *

data = [DataLoader(split="BioASQ-task11bPhaseB-testset1.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset2.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset3.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset4.json")]

zsl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting"
zsl_dst_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting"

for results_file in os.listdir(zsl_dir):
    results = {'questions': []}
    generations = read_dict(zsl_dir, results_file)
    bioasq_data = data[int(results_file.split("test")[1].split("_")[0])-1]
    for question in bioasq_data.data['questions']:
        results['questions'].append({'type': question['type'], 'ideal_answer': generations[question['id']],
                                     'id': question['id']})
    write_dict(results, zsl_dst_dir, "baseline_" + results_file)

icl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\few_shot_prompting"
icl_dst_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\few_shot_prompting"

for results_file in os.listdir(icl_dir):
    results = {'questions': []}
    generations = read_dict(icl_dir, results_file)
    bioasq_data = data[int(results_file.split("test")[1].split("_")[0])-1]
    for question in bioasq_data.data['questions']:
        results['questions'].append({'type': question['type'], 'ideal_answer': generations[question['id']],
                                     'id': question['id']})
    write_dict(results, icl_dst_dir, "baseline_" + results_file)

print("Results were reported successfully!")
