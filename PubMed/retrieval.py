import os
import json
import numpy as np
import torch
import sys

from PubMed.BioASQ import *
from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import pipeline, set_seed, BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel, GPT2Tokenizer

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


def model1():
    data = DataLoader()
    model = TransformerLanguageModel.from_pretrained(
        "/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
        "/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/checkpoints/Pre-trained-BioGPT-Large/checkpoint.pt",
        data="/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
        tokenizer='moses',
        bpe='fastbpe',
        bpe_codes="data/biogpt_large_bpecodes",
        min_len=100,
        max_len_b=1024)
    model.cuda()

    for question in data.data['questions']:
        src_tokens = model.encode(question['body'])
        generate = model.generate([src_tokens], beam=5)[0]
        output = model.decode(generate[0]["tokens"])
        question['BioGPT_answer'] = output

    write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
               "BioGPT_Large_checkpoint.txt")
    completed = True


def model2():
    set_seeds(0)
    # data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset2.json")
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
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
        inputs = tokenizer.encode(LLM_query, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(inputs, max_length=len(LLM_query))
            output = tokenizer.decode(model_generation[0])
        except:
            output = ""
        answers[question['id']] = output

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_test1.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_prompting",
               "BioGPT_Large_HF_test2.txt")
    completed = True


def model3():
    set_seeds(0)
    # data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset4.json")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = model.to(device)

    answers = {}
    few_shot_prompting = False

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query = question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']

        # Perform a query to the Large Language Model
        inputs = tokenizer.encode(LLM_query, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(inputs, max_length=len(LLM_query))
            output = tokenizer.decode(model_generation[0])
        except:
            output = ""
        answers[question['id']] = output

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_PubMed_checkpoint.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/zero_shot_prompting",
               "BioGPT_Large_HF_PubMed_test4b.txt")
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
        inputs = tokenizer.encode(LLM_query, return_tensors="pt").to(device)
        model_generation = model.generate(inputs, max_length=len(LLM_query))
        output = tokenizer.decode(model_generation[0])
        answers[question['id']] = output

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_test1.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_prompting",
               "PubMedGPT_test4.txt")
    completed = True


def model2_icl():
    set_seeds(0)
    train_data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset2.json")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = model.to(device)

    answers = {}
    few_shot_prompting = True
    train_examples = 1
    LLM_query = ""

    for question in train_data.data['questions'][0:train_examples]:
        # Construct the question to the Large Language Model
        LLM_query += "\n" + question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']
        LLM_query += "\nAnswer:" + question['ideal_answer'][0]

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query += "\n" + question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']
        LLM_query += "\nAnswer:"

        # Perform a query to the Large Language Model
        inputs = tokenizer.encode(LLM_query, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(inputs, max_length=len(LLM_query))
            output = tokenizer.decode(model_generation[0])
        except:
            output = ""
        answers[question['id']] = output

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_test1.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/few_shot_in_context_learning",
               "BioGPT_Large_HF_test2.txt")
    completed = True


def model3_icl():
    set_seeds(0)
    train_data = DataLoader()
    data = DataLoader(split="BioASQ-task11bPhaseB-testset4.json")
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = model.to(device)

    answers = {}
    few_shot_prompting = False
    train_examples = 1
    LLM_query = ""

    for question in train_data.data['questions'][0:train_examples]:
        # Construct the question to the Large Language Model
        LLM_query += "\n" + question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']
        LLM_query += "\nAnswer:" + question['ideal_answer'][0]

    for question in data.data['questions']:
        # Construct the question to the Large Language Model
        LLM_query += "\n" + question['body']
        if few_shot_prompting:
            LLM_query += "\nHere is some useful additional information:"
            for snippet in question['snippets']:
                LLM_query += "\n" + snippet['text']
        LLM_query += "\nAnswer:"

        # Perform a query to the Large Language Model
        inputs = tokenizer.encode(LLM_query, return_tensors="pt").to(device)
        try:
            model_generation = model.generate(inputs, max_length=len(LLM_query))
            output = tokenizer.decode(model_generation[0])
        except:
            output = ""
        answers[question['id']] = output

    # write_list(data.data['questions'], "/media/geomos/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels",
    #            "BioGPT_Large_HF_PubMed_checkpoint.txt")
    write_dict(answers, "/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_results/zero_shot_in_context_learning",
               "BioGPT_Large_HF_PubMed_test4.txt")
    completed = True

model4()
