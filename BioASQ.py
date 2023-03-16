import os
import json
import subprocess
import numpy as np
import torch
import sys

from tqdm import tqdm
from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import pipeline, set_seed, BioGptTokenizer, BioGptForCausalLM


def set_seeds(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


class DataLoader:
    def __init__(self, mode="Load"):
        # Set the appropriate paths and load the data .json files
        # TODO: Try running crawler.sh through Terminal before the first time executing this script (changing paths).
        # TODO: If that fails set the paths corresponding to your PC and local_dump="Download" and run one time.
        # TODO: If the download process completes partially set local_dump="Partial Download" and run again.
        # TODO: (Disclaimer) This process will take one day to complete thus avoid is possible.
        self.BioASQ_DIR = r"/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/"
        self.PubMed_DIR = r"/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/PubMed"
        self.FAISS_DIR = r"/media/georg_mosh/Data SSD/BIOMEDICAL/checkpoints/PubMed_index_16_8_100_10_10-4"
        self.DPR_DIR = r"/media/georg_mosh/Data SSD/BIOMEDICAL/checkpoints/PubMed_DPR_16_8_100_10_10-4"
        self.data = self.load_json()

        # Get a local copy of the corresponding publications in the right format
        self.websites, self.missing_websites = self.collect_data(mode)

    def collect_data(self, mode):
        if mode in {"Download", "Partial Download", "Load"}:
            if "Download" in mode:
                collected = self.crawler_pubmed(mode="standard") if "Partial" not in mode \
                            else self.crawler_pubmed(mode="failure")
            else: collected = self.collect_websites(mode="standard")
            missing = self.collect_websites(mode="failure")
            print("Total files required: " + str(len(collected)) + "\nMissing files: " + str(len(missing)))
        else:
            raise Exception("The supported states are Complete or Partial Download and Load.")

        return collected, missing

    def load_json(self):
        file = open(os.path.join(self.BioASQ_DIR, "training11b.json"))
        data = json.load(file)
        file.close()

        return data

    def load_text(self, filename):
        file = open(os.path.join(self.PubMed_DIR, filename))
        data = file.read()
        file.close()

        return data

    def write_script(self, filename, context):
        filehandle = open(filename, 'w')
        filehandle.write(context)
        filehandle.close()

    def collect_websites(self, mode):
        if mode == "standard":
            websites = set([])
            for question in self.data["questions"]:
                for document in question["documents"]:
                    websites.add(document)
        elif mode == "failure":
            websites = set([])
            for question in self.data["questions"]:
                for document in question["documents"]:
                    website = document.split("/")[-1]
                    if not os.path.exists(os.path.join(self.PubMed_DIR, website)):
                        websites.add(document)
        else:
            raise Exception("The supported running modes are standard and failure (if files are missing).")

        return websites

    def crawler_pubmed(self, mode, format="pubmed"):
        # Collect all the websites of the corresponding publications.
        websites = self.collect_websites(mode)

        # Shorten websites' URLs to make crawling feasible (allow downloading specific format).
        # TODO: Install client for Firebase by running "pip install python-firebase-url-shortener" in Terminal.
        # TODO: (Optionally) You may want to first create a virtual environment.
        naming_conventions = {}
        from firebase import UrlShortener
        link_generator = UrlShortener()
        websites = list(websites)
        for i in tqdm(range(len(websites))):
            website = websites[i]
            naming_conventions[website.split("/")[-1]] = link_generator.shorten(website + "/?format=" + format)

        # Add websites' URLs (shortened versions) to the bash script using wget commands.
        # Apply the naming conversions to the downloaded websites by renaming the files as in PubMed.
        # Store locally the corresponding bash script to run.
        bash_script = "#! /bin/bash\ncd \"" + self.PubMed_DIR + "\""
        doc_IDs = list(naming_conventions.keys())
        for i in tqdm(range(len(doc_IDs))):
            doc_ID = doc_IDs[i]
            bash_script += "\nwget " + naming_conventions[doc_ID]
            bash_script += "\nmv " + naming_conventions[doc_ID].split("/")[-1] + " " + doc_ID
        self.write_script("../crawler.sh", bash_script)

        # Subprocess call only tested in Ubuntu 20.04 - should work in all Linux versions.
        # This should also work on Mac-OS (as is or with minor changes).
        subprocess.call("../crawler.sh")


"""
Write a given list.
"""
def write_list(list, directory, filename, save=True):
    if save:
        with open(os.path.join(directory, filename), 'w') as handle:
            json.dump(list, handle)

def model1():
    data = DataLoader()
    model = TransformerLanguageModel.from_pretrained(
        "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
        "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioGPT/checkpoints/Pre-trained-BioGPT-Large/checkpoint.pt",
        data="/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
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

    write_list(data.data['questions'], "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels", "BioGPT_Large_checkpoint.txt")
    completed = True


def model2():
    set_seeds(0)
    data = DataLoader()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = model.to(device)

    for question in data.data['questions']:
        inputs = tokenizer.encode(question['body'], return_tensors="pt").to(device)
        model_generation = model.generate(inputs, max_length=1024)
        output = tokenizer.decode(model_generation[0])
        question['BioGPT_HF_answer'] = output

    write_list(data.data['questions'], "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels", "BioGPT_Large_HF_checkpoint.txt")
    completed = True


def model3():
    set_seeds(0)
    data = DataLoader()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = model.to(device)

    for question in data.data['questions']:
        inputs = tokenizer.encode(question['body'], return_tensors="pt").to(device)
        model_generation = model.generate(inputs, max_length=1024)
        output = tokenizer.decode(model_generation[0])
        question['BioGPT_HF_QA_answer'] = output

    write_list(data.data['questions'], "/media/georg_mosh/Data SSD/AUEB BIOMEDICAL DATA/BioASQ11/LargeLanguageModels", "BioGPT_Large_HF_PubMed_checkpoint.txt")
    completed = True


model3()
x=2
