import os
import json
import numpy as np
import torch
import sys

from PubMed.BioASQ import *
from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer, BioGptTokenizer, BioGptForCausalLM

def set_seeds(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


"""
Write a given list.
"""
def write_list(list, directory, filename, save=True):
    if save:
        with open(os.path.join(directory, filename), 'w') as handle:
            json.dump(list, handle)


"""
Write a given dictionary.
"""
def write_dict(dict, directory, filename):
    with open(os.path.join(directory, filename), 'w') as convert_file:
        convert_file.write(json.dumps(dict))

def model():
    set_seeds(0)
    data = DataLoader()
    stop = True

def model2():
    set_seeds(0)
    # data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset4.json")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = model.to(device)

    answers = {}
    few_shot_prompting = True

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query = question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']

        # Perform a query to the Large Language Model
        inputs = tokenizer(LLM_query, truncation=True, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(
                input_ids=inputs.input_ids.to(device, non_blocking=True),
                attention_mask=inputs.attention_mask.to(device, non_blocking=True),
                max_length=550,  # type: ignore
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True, )
            degeneration = tokenizer.batch_decode(model_generation.sequences, skip_special_tokens=True)[0]
            answers[question['id']] = degeneration
        except:
            answers[question['id']] = ''

            # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_test1.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_prompting",
               "BioGPT_Large_HF_test4_bs5.txt")
    completed = True


def model3():
    set_seeds(0)
    # data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset4.json")
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = model.to(device)

    answers = {}
    few_shot_prompting = True

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query = question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']

        # Perform a query to the Large Language Model
        inputs = tokenizer(LLM_query, truncation=True, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(
                input_ids=inputs.input_ids.to(device, non_blocking=True),
                attention_mask=inputs.attention_mask.to(device, non_blocking=True),
                max_length=550,  # type: ignore
                num_beams=1,
                num_return_sequences=1,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True, )
            degeneration = tokenizer.batch_decode(model_generation.sequences, skip_special_tokens=True)[0]
            answers[question['id']] = degeneration
        except:
            answers[question['id']] = ''

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_PubMed_checkpoint.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_prompting",
               "BioGPT_Large_HF_PubMed_test4_bs1.txt")
    completed = True


def model4():
    set_seeds(0)
    # data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset4.json")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")
    model = model.to(device)

    answers = {}
    few_shot_prompting = True

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query = question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']

        # Perform a query to the Large Language Model
        inputs = tokenizer(LLM_query, truncation=True, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(
                input_ids=inputs.input_ids.to(device, non_blocking=True),
                attention_mask=inputs.attention_mask.to(device, non_blocking=True),
                max_length=550,  # type: ignore
                num_beams=1,
                num_return_sequences=1,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,)
            degeneration = tokenizer.batch_decode(model_generation.sequences, skip_special_tokens=True)[0]
            answers[question['id']] = degeneration
        except:
            answers[question['id']] = ''

    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_prompting",
               "PubMedGPT_test4_bs1.txt")
    completed = True

model()
