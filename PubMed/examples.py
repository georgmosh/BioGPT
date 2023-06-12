import torch
import numpy as np
import sys

from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import pipeline, set_seed, BioGptTokenizer, BioGptForCausalLM

print("User Current Version:-", sys.version)


def set_seeds(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def load_state_dict(model, pubmed_biogpt_ckp):
    state_dict = torch.load(pubmed_biogpt_ckp)['model']
    state_dict_v2 = {}
    state_dict_v2['_float_tensor'] = torch.FloatTensor([0])
    for key in state_dict.keys():
        state_dict_v2["models.0." + key] = state_dict[key]
    state_dict_v2['models.0.decoder.embed_tokens.weight'] = state_dict_v2['models.0.decoder.embed_tokens.weight'][:57717, :]             # todo: load fine-tuned model properly!
    state_dict_v2['models.0.decoder.output_projection.weight'] = state_dict_v2['models.0.decoder.output_projection.weight'][:57717, :]

    model.load_state_dict(state_dict_v2)


if __name__ == "__main__":
    set_seeds(0)
    # m = TransformerLanguageModel.from_pretrained(
    #     "/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
    #     "/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/checkpoints/Pre-trained-BioGPT-Large/checkpoint.pt",
    #     data="/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/data/BioGPT-Large",
    #     tokenizer='moses',
    #     bpe='fastbpe',
    #     bpe_codes="data/biogpt_large_bpecodes",
    #     min_len=100,
    #     max_len_b=1024)
    # m.cuda()
    # src_tokens = m.encode("COVID-19 is")
    # generate = m.generate([src_tokens], beam=5)[0]
    # output = m.decode(generate[0]["tokens"])
    # print(output)
    #
    # src_tokens = m.encode("The Effect of chloroquine on cultured fibroblasts is")
    # generate = m.generate([src_tokens], beam=5)[0]
    # output = m.decode(generate[0]["tokens"])
    # print(output)
    #
    # load_state_dict(m, "/media/geomos/AUEB BIOMEDICAL DATA/BioGPT/checkpoints/QA-PubMed-BioGPT-Large/checkpoint_avg.pt")
    #
    # m.cuda()
    # src_tokens = m.encode("The Effect of chloroquine on cultured fibroblasts is")
    # generate = m.generate([src_tokens], beam=5)[0]
    # output = m.decode(generate[0]["tokens"])
    # print(output)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = model.to(device)

    inputs = tokenizer.encode('COVID-19 is', return_tensors="pt").to(device)
    model_generation = model.generate(inputs, max_length=100)
    output = tokenizer.decode(model_generation[0], skip_special_tokens=True)
    print(output)

    inputs = tokenizer.encode('The Effect of chloroquine on cultured fibroblasts is', return_tensors="pt").to(device)
    model_generation = model.generate(inputs, max_length=100)
    output = tokenizer.decode(model_generation[0], skip_special_tokens=True)
    print(output)

    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = model.to(device)

    inputs = tokenizer.encode('COVID-19 is', return_tensors="pt").to(device)
    model_generation = model.generate(inputs, max_length=100)
    output = tokenizer.decode(model_generation[0])
    print(output)

    inputs = tokenizer.encode('The Effect of chloroquine on cultured fibroblasts is', return_tensors="pt").to(device)
    model_generation = model.generate(inputs, max_length=100)
    output = tokenizer.decode(model_generation[0])
    print(output)
