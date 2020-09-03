import argparse
import json
import os
import random
import re

import numpy as np
from tqdm import tqdm

from baselines.textseg import text_manipulation, wiki_thresholds
from baselines.textseg import wiki_utils

"""
Generates training data for the textseg algorithm, dividing all the paragraphs into sentences 
"""


class ConsecutiveDocumentsTextSeg(object):
    def __init__(self, files, input_folder, data_dir="./data_og_textseg"):
        """
        Essentially bucket-sort all inputs according to their labels.
        Then we can simply "draw" from the correct bucket
        :param files: files to process
        :param data_dir: where to store the output
        :param input_folder: the folder where the files are located

        """
        # Each document is now a dictionary of sections and
        # list of paragraphs in the section, all_docs contains all the documents
        self.all_docs = []
        self.all_lens = {}
        self.num_labels = None
        self.final_data = []

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        for f in tqdm(sorted(files)):
            doc = {}
            with open(os.path.join(input_folder, f)) as fp:
                tos = json.load(fp)
            # Now equal to the test set
            if len(tos["level1_headings"]) < 2:
                continue
            for section in tos["level1_headings"]:
                # Transform dict into X/y sample
                text = section["text"]
                label = section["section"]
                doc = self.add_to_section(text, label, doc)
            self.all_docs.append(doc)

    @staticmethod
    def add_to_section(section_text, section_label, doc):
        processed_text = section_text

        if section_label in doc:
            doc[section_label] += processed_text
        else:
            doc[section_label] = processed_text
        return doc

    @staticmethod
    def count_str_occurrences(string, findStr):
        return len(string.split(findStr)) - 1

    def process_section(self, section, idx):
        global num_sentences_for_avg
        global sum_sentences_for_avg
        num_sentences_for_avg = 0
        sum_sentences_for_avg = 0
        sentences = text_manipulation.split_sentences(section, idx)
        section_sentences = []
        num_sentences = 0
        num_formulas = 0
        num_codes = 0
        for sentence in sentences:

            sentence_words = text_manipulation.extract_sentence_words(sentence)
            if len(sentence_words) < wiki_thresholds.min_words_in_sentence:
                # ignore this sentence
                continue
            sum_sentences_for_avg += len(sentence_words)
            num_sentences_for_avg += 1

            num_formulas += self.count_str_occurrences(sentence,
                                                       wiki_utils.get_formula_token())
            num_codes += self.count_str_occurrences(sentence,
                                                    wiki_utils.get_codesnipet_token())
            num_sentences += 1
            section_sentences.append(sentence)

        valid_section = True
        error_message = None
        if num_sentences < wiki_thresholds.min_sentence_in_section:
            valid_section = False
            error_message = "Sentence count in section is too low!"

        section_text = ''.join(section_sentences)
        if len(re.findall('[a-zA-Z]', section_text)) < \
                wiki_thresholds.min_section_char_count:
            valid_section = False
            error_message = "Char count in section is too low!"

        if num_formulas >= wiki_thresholds.max_section_formulas_count:
            valid_section = False
            error_message = f"Number of formulas in section is too high: {num_formulas}"

        if num_codes >= wiki_thresholds.max_section_code_snipet_count:
            valid_section = False
            error_message = f"Number of code snippets in section is too high: " \
                            f"{num_codes}"

        return valid_section, section_sentences, error_message

    def generate_data(self, foldername):
        """
        :param foldername: Storage folder name (train, test, dev)
        :return: None
        """
        os.makedirs(os.path.join(self.data_dir, foldername), exist_ok=True)
        print(f"Generating {foldername} data...")
        start_id = 0
        for doc_id, doc in enumerate(tqdm(self.all_docs), start=start_id):
            with open(os.path.join(*[self.data_dir, foldername, str(doc_id)]), "wb") as f:
                for index, (label, sections) in enumerate(doc.items()):
                    is_valid_section, section_sentences, message = self.process_section(sections, label)
                    if is_valid_section:
                        f.write(("========," + str(index) + "," + label + "." + "\n").encode('utf-8'))
                        f.write(("\n".join(section_sentences) + "\n").encode('utf-8'), )

        # after preprocessing some files will be empty, go back and remove them
        files = sorted(os.path.join(self.data_dir, foldername))
        for f in tqdm(sorted(files)):
            with open(os.path.join(*[self.data_dir, foldername, f])) as fp:
                raw_content = fp.read()
                sections = [s for s in raw_content.strip().split("\n") if len(s) > 0 and s != "\n"]
            if len(sections) < 1:
                os.remove(os.path.join(*[self.data_dir, foldername, f]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Processing the terms of services dataset and spliting them by '
                    'sentences to match the input data for textseg.')
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the raw data, '
                             'keep in mind the preprocessing steps. ',
                        default="./tos-data-og")
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="../training_data_textseg")
    parser.add_argument('--heading_level', help='first or second level headings',
                        type=int, default=1)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    heading_level = args.heading_level  # first or second level headings

    train_fraction = 0.8
    dev_fraction = 0.1
    test_fraction = 0.1

    if train_fraction + dev_fraction + test_fraction != 1.0:
        raise ValueError("Fractions must add up to 1.0!")
    files = sorted(os.listdir(input_folder))

    np.random.seed(69120)
    file_order = np.random.choice(files, len(files), replace=False)
    train_files = file_order[:int(len(files) * train_fraction)]
    dev_files = file_order[int(len(files) * train_fraction):
                           int(len(files) * (train_fraction + dev_fraction))]
    test_files = file_order[int(len(files) * (train_fraction + dev_fraction)):]
    # hack to accustom to eirene og-test, where three files were missing
    test_files = [el for el in test_files if not el.startswith('data')]

    random.seed(12)

    train_docs = ConsecutiveDocumentsTextSeg(train_files, input_folder=input_folder,
                                             data_dir=output_folder)
    test_docs = ConsecutiveDocumentsTextSeg(test_files, input_folder=input_folder,
                                            data_dir=output_folder)
    dev_docs = ConsecutiveDocumentsTextSeg(dev_files, input_folder=input_folder,
                                           data_dir=output_folder)

    train_docs.generate_data("train")
    test_docs.generate_data("test")
    dev_docs.generate_data("dev")

    print("Done!")
