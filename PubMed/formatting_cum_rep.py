from helpers import *
from PubMed.BioASQ import *

data = [DataLoader(split="BioASQ-task11bPhaseB-testset1.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset2.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset3.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset4.json")]

initial_models = ["PubMedGPT_test1_bs3.txt", "PubMedGPT_test2_bs3.txt", "PubMedGPT_test3_bs3.txt",
                  "PubMedGPT_test4_bs3.txt"]
supplementary_models = ["PubMedGPT_test1_bs5.txt", "PubMedGPT_test2_bs5.txt", "PubMedGPT_test3_bs5.txt",
                        "PubMedGPT_test4_bs5.txt"]

assert len(data) == len(initial_models)
assert len(data) == len(supplementary_models)

zsl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting"
zsl_dst_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_ensembling"

results = []
for idx in range(len(data)):
    results_file = initial_models[idx]
    batch_results = {'questions': []}
    generations = read_dict(zsl_dir, results_file)
    bioasq_data = data[int(results_file.split("test")[1].split("_")[0])-1]
    for question in bioasq_data.data['questions']:
        answer = generations[question['id']] \
                 if len(re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', re.sub(question['body'].lower(), '',
                               generations[question['id']].lower()))) > 10 else None
        answer_metadata = {'type': question['type'], 'ideal_answer': answer, 'id': question['id']}
        if question['type'] == 'yesno':
            answer_metadata['exact_answer'] = "yes"
        elif question['type'] == 'factoid':
            answer_metadata['exact_answer'] = [[""]]
        batch_results['questions'].append(answer_metadata)
    results.append(batch_results)

    results_file = initial_models[idx]
    batch_results = {'questions': []}
    generations = read_dict(zsl_dir, results_file)
    for question in results[-1]['questions']:
        if question['ideal_answer'] is None:
            question['ideal_answer'] = generations[question['id']]

    write_dict(results, zsl_dst_dir, "baseline_" + results_file)
