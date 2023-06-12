import os
import json
import subprocess


from tqdm import tqdm

class DataLoader:
    def __init__(self, mode="Load", split="training11b.json"):
        # Set the appropriate paths and load the data .json files
        # TODO: Try running crawler.sh through Terminal before the first time executing this script (changing paths).
        # TODO: If that fails set the paths corresponding to your PC and local_dump="Download" and run one time.
        # TODO: If the download process completes partially set local_dump="Partial Download" and run again.
        # TODO: (Disclaimer) This process will take one day to complete thus avoid is possible.
        self.BioASQ_DIR = r"/media/geomos/AUEB BIOMEDICAL SYSTEMS/BioASQ11_data"
        self.PubMed_DIR = r"/media/geomos/AUEB BIOMEDICAL SYSTEMS/PubMed"
        self.FAISS_DIR = r"/media/geomos/BIOMEDICAL/checkpoints/PubMed_index_16_8_100_10_10-4"
        self.DPR_DIR = r"/media/geomos/BIOMEDICAL/checkpoints/PubMed_DPR_16_8_100_10_10-4"
        self.data = self.load_json(split)

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

    def load_json(self, filename):
        file = open(os.path.join(self.BioASQ_DIR, filename))
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