import os
import json
import re

"""
Read a results dictionary.
"""
def read_dict(directory, filename):
    with open(os.path.join(directory, filename)) as handle:
        data = handle.read()
    dictionary = json.loads(data)

    return dictionary

"""
Add an element to a dictionary
"""
def add_to_dict(dictionary, element, key):
    if not key in dictionary.keys():
        dictionary[key] = []
    dictionary[key].append(element)

    return dictionary

"""
Write a given dictionary.
"""
def write_dict(dict, directory, filename):
    with open(os.path.join(directory, filename), 'w') as convert_file:
        convert_file.write(json.dumps(dict))

"""
Write a given list.
"""
def write_list(list, directory, filename, save=True):
    if save:
        with open(os.path.join(directory, filename), 'w') as handle:
            json.dump(list, handle)

"""
Write a given list one element per line.
"""
def write_entity_list(list, directory, filename):
    entities = ""
    for elem in list: entities += (elem + "\n")
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as handle:
        handle.write(entities)

"""
Read a given list.
"""
def read_list(directory, filename):
    with open(os.path.join(directory, filename), 'r') as handle:
        list = json.load(handle)

    return list

"""
Write a given text.
"""
def write(text, path=None, filename=None, write_back=True, encoding='utf-8'):
    # Write preprocessing results back to file.
    if write_back:
        with open(os.path.join(path, filename), mode='w', encoding=encoding) as file:
            file.write(text)

    return text

"""
Apply basic preprocessing to a bunch of text.
"""
def clean_binary(text):
    if text.lower().startswith("yes"):
        return re.sub(" +", " ", re.sub(r'[^A-Za-z0-9()\[\] ]+', ' ', text[3:])).strip()
    elif text.lower().startswith("no"):
        return re.sub(" +", " ", re.sub(r'[^A-Za-z0-9()\[\] ]+', ' ', text[2:])).strip()
    else:
        return re.sub(" +", " ", re.sub(r'[^A-Za-z0-9()\[\] ]+', ' ', text)).strip()

"""
Initialize container for traditional retrieval evaluation.
"""
def initialize_eval(dim):
    res_per_class = []
    for i in range(dim):
        res_per_class.append({"TP": 0, "FP": 0, "TN": 0, "FN": 0, "Precision": 0, "Recall": 0, "F1": 0}.copy())

    return res_per_class

"""
Serialize scores' numerical values for writing to disk.
"""
def serialize_float32_scores(scored_premises):
    str_scored_premises = {}
    for premise in scored_premises.keys():
        str_scored_premises[premise] = str(scored_premises[premise])

    return str_scored_premises

"""
Serialize scores' numerical values for writing to disk.
"""
def serialize_float32_probs(prob_premises):
    str_prob_premises = []
    for prob in prob_premises:
        str_prob_premises.append(str(prob))

    return str_prob_premises

"""
Read file to obtain its raw string contents.
"""
def read_file(src_directory, filename, encoding=None, log=True):
    if log: print("Reading file:", filename)
    with open(os.path.join(src_directory, filename), mode='r', encoding=encoding) as file:
        content = file.read()
    return content

"""
Copy file to another path for batch processing.
"""
def process_file(src_directory, dest_directory, filename, encoding, batch_count):
    print("Processing file:", filename)
    content = process_file(src_directory, filename, encoding, log=False)
    destination_path = os.path.join(dest_directory, "B" + str(batch_count))
    _ = write(content, destination_path, filename)

"""
Get the raw text files' names for a given container.
"""
def get_filename(container):
    new_container = []
    for file in container:
        new_container.append(file + ".txt")

    return new_container
