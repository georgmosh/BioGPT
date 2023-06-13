from helpers import *
from PubMed.BioASQ import *
from evaluation import extract_snippets

data = [DataLoader(split="BioASQ-task11bPhaseB-testset1.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset2.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset3.json"),
        DataLoader(split="BioASQ-task11bPhaseB-testset4.json")]

zsl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting"
zsl_dst_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\zero_shot_prompting_processed"

for results_file in os.listdir(zsl_dir):
    results = {}
    generations = read_dict(zsl_dir, results_file)
    bioasq_data = data[int(results_file.split("test")[1].split("_")[0])-1]
    gold_relevant_snippets = extract_snippets(bioasq_data.data['questions'])
    for question in bioasq_data.data['questions']:
        try:
            processed = re.sub(re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', question['body'].lower()), '',
                               re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', generations[question['id']].lower()))
            processed = re.sub(re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '',
                                      gold_relevant_snippets[question['id']].lower()), '',
                               re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', processed))
        except:
            processed = question['body'].lower()
        results[question['id']] = processed
    write_dict(results, zsl_dst_dir, results_file)

icl_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\few_shot_prompting"
icl_dst_dir = r"D:\AUEB Material\NSRC Demokritos - BioASQ\BioASQ11_results\few_shot_prompting_processed"

for results_file in os.listdir(icl_dir):
    results = {}
    generations = read_dict(icl_dir, results_file)
    bioasq_data = data[int(results_file.split("test")[1].split("_")[0])-1]
    gold_relevant_snippets = extract_snippets(bioasq_data.data['questions'])
    for question in bioasq_data.data['questions']:
        try:
            processed = re.sub(re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', question['body'].lower()), '',
                               re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', generations[question['id']].lower()))
            processed = re.sub(re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '',
                                      gold_relevant_snippets[question['id']].lower()), '',
                               re.sub(r'[^A-Za-zΑ-Ω0-9α-ωά-ώάέήίόύώϊΐϋΰ ]+', '', processed))
        except:
            processed = question['body'].lower()
        results[question['id']] = processed
    write_dict(results, icl_dst_dir, results_file)

print("Results were reported successfully!")
