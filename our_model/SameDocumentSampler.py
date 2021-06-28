"""
Reads each section in a file and samples paragraphs from the same section as positive samples,
and paragraphs from a different section in the same document as a negative sample.
"""

from transformers import BertTokenizer, RobertaTokenizer
from tokenizers import SentencePieceBPETokenizer
from shutil import copyfile
from tqdm import tqdm

import pandas as pd
import numpy as np
import argparse
import random
import json
import os


class SameDocumentSampler(object):
    def __init__(self, files, folder, storage_method="raw", force_shorten=True,
                 data_dir="./data_og_consecutive", tokenizer_path="./"):
        """
        Make a consecutive document structure and draw samples where the positive
        examples come from the paragraphs of the same section in text and
        negative examples from different sections
        :param files: files to process
        :param storage_method: Either "raw" (remove \n and \t),
                               "bert" (BERT tokenization),
                               "roberta" (RoBERTa tokenization), or
                               "token" (own tokenizer)
        :param force_shorten: BERT-based models have length limitation.
        :param data_dir: where to store the output
        :param tokenizer_path: where to find the sentence piece tokenizer

        """
        self.training_pairs = []
        self.training_triplets = []
        self.tokenizer_path = tokenizer_path

        self.storage_method = storage_method
        self.force_shorten = force_shorten
        self.tokenizer = None
        self.set_tokenizer()

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.docs = []

        for f in tqdm(files):
            doc = {}
            with open(os.path.join(folder, f)) as fp:
                tos = json.load(fp)
            for section in tos:
                # Transform dict into X/y sample
                text = section["Text"]
                label = section["Section"]
                doc = self.add_to_section(text, label, doc)

            self.docs.append(doc)

    def set_tokenizer(self):
        if self.storage_method == "raw":
            pass  # Essentially keep it None. Important for exceptions
        elif self.storage_method == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.storage_method == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        elif self.storage_method == "token":
            self.tokenizer = SentencePieceBPETokenizer(os.path.join(self.tokenizer_path, "/vocab.json"),
                                                       os.path.join(self.tokenizer_path, "merges.txt"))
        else:
            raise ValueError("Unknown storage method encountered!")

    def add_to_section(self, section_text, section_heading, doc):
        processed_text = self.transform_text_accordingly(section_text)

        if section_heading in doc:
            doc[section_heading].append(processed_text)
        else:
            doc[section_heading] = [processed_text]
        return doc

    def transform_text_accordingly(self, text):
        if self.storage_method == "raw":
            return text
        elif self.storage_method == "bert" or self.storage_method == "roberta":
            encoded_text = self.tokenizer.encode(text)
            # Shorten if necessary
            if self.force_shorten and len(encoded_text) > 512:
                # Still need the last token
                return encoded_text[:511] + [encoded_text[-1]]
            else:
                return encoded_text
        else:  # the case for custom tokenizer
            # TODO: Currently doesn't support longer input formats due to the hard-coded cutoff
            encoded_text = self.tokenizer.encode(text).ids
            if self.force_shorten and len(encoded_text) > 512:
                return encoded_text[:512]  # Own encoding has no special symbols
            else:
                return encoded_text

    def generate_data(self, sample_number, file_name):
        """
        :param sample_number: Number of positive/negative samples, respectively.
            A document needs to have at least two sections, each with num_samples samples.
        :param file_name: Storage file name (train, test, dev)
        :return: None
        """
        docs_skipped = 0
        print(f"Generating {file_name} data...")
        for doc in tqdm(self.docs):
            for heading, sections in doc.items():
                if len(doc) < 2:
                    docs_skipped += 1
                    continue
                for section in sections:
                    # Ensure we only sample other paragraphs in the same section
                    other_sections = list(sections)
                    other_sections.remove(section)

                    # Limits the number of samples to either sample_number or the number of available other paragraphs
                    # in the same section. This avoids "over-sampling", especially since we repeat this process for
                    # each paragraph in the section
                    # TODO: Evaluate effect of the normalization with // 2
                    positive_sections = np.random.choice(other_sections,
                                                         min(sample_number, len(other_sections) // 2),
                                                         replace=False)

                    for section_pos in positive_sections:
                        self.training_pairs.append({"section_1": section,
                                                    "section_2": section_pos,
                                                    "label": 1})
                        # Generate a matching pair of a negative sample to keep the training set balanced.
                        section_neg = self.generate_negative_sample(heading, section, doc)

                        self._add_samples_triplet_loss(section, section_pos, section_neg)

        print(f"{len(self.training_pairs)} sample generated.\n"        
              f"{docs_skipped} docs skipped entirely ({docs_skipped/len(self.docs):.2f}% of total number of docs)")
        print("writing data to tsv file...")
        df_data = pd.DataFrame(self.training_pairs, columns=["section_1", "section_2", "label"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '.tsv'),
                       sep='\t', index=False, header=False)
        df_data = pd.DataFrame(self.training_triplets,
                               columns=["section_center", "section_pos", "section_neg"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '_triplet.tsv'),
                       sep='\t', index=False, header=False)

    def generate_negative_sample(self, heading, section, doc):
        # choose a random other section in the document
        rand_other_heading = heading
        while rand_other_heading == heading:
            rand_other_heading = random.choice(list(doc.keys()))
        rand_pos = random.randint(0, len(doc[rand_other_heading]) - 1)
        negative_section = doc[rand_other_heading][rand_pos]
        self.training_pairs.append({"section_1": section,
                                    "section_2": negative_section,
                                    "label": 0})
        return negative_section

    def _add_samples_triplet_loss(self, section_pos, section_neg, section):
        self.training_triplets.append({"section_center": section,
                                       "section_pos": section_pos,
                                       "section_neg": section_neg})


def load_files(files, folder):
    paragraphs = []
    section_headings = []
    for f in tqdm(files):
        with open(os.path.join(folder, f)) as fp:
            tos = json.load(fp)

        for section in tos:
            # Transform dict into X/y sample
            text = section["Text"]
            title = section["Section"]

            paragraphs.append(text)
            section_headings.append(title)

    return paragraphs, section_headings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and create training samples from documents.'
                    'This utilizes the consecutive training pair strategy, which is the best-performing one.')
    parser.add_argument('--input-folder',
                        help='The location of the folder containing the raw data, '
                             'keep in mind the preprocessing steps.',
                        default="../resources/tos-data-og")
    parser.add_argument('--output-folder', help='The location for the output folder ',
                        default="../resources/training_data")
    parser.add_argument('--sentence-tokenizer-path',
                        help='Path to the sentence tokenizer folder, '
                             'set this if you are using the storage_method=token.',
                        default="./")
    parser.add_argument('--sample-number',
                        help='Number of positive and negative samples to be created '
                             'from each section or paragraph', type=int, default=3)
    parser.add_argument('--storage-method',
                        help='Either "raw",  "bert" (BERT tokenization), '
                             '"roberta" (RoBERTa tokenization), or "token" (own tokenizer)',
                        choices=['raw', 'roberta', 'bert', 'token'], default="raw")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    sample_number = args.sample_number  # number of positive and negative samples to be create from each section
    storage_method = args.storage_method  # Which method to store?
    tokenizer_path = args.sentence_tokenizer_path

    all_inputs = []
    all_labels = []

    train_fraction = 0.8
    dev_fraction = 0.1
    test_fraction = 0.1
    if train_fraction + dev_fraction + test_fraction != 1.0:
        raise ValueError("Fractions must add up to 1.0!")
    files = sorted(os.listdir(input_folder))
    np.random.seed(69120)
    file_order = np.random.choice(files, len(files), replace=False)
    train_files = file_order[:int(len(files) * train_fraction)]
    dev_files = file_order[int(len(files) * train_fraction): int(len(files) * (train_fraction + dev_fraction))]
    test_files = file_order[int(len(files) * (train_fraction + dev_fraction)):]

    input_train, label_train = load_files(train_files, input_folder)
    input_dev, label_dev = load_files(dev_files, input_folder)
    input_test, label_test = load_files(test_files, input_folder)

    dst_folder = "og-test"  # copy the files from the test set to the new location
    # for text segmentation we need the original structure of the files in the test set.
    os.makedirs(os.path.join(output_folder, dst_folder), exist_ok=True)
    for file in test_files:
        src = os.path.join(input_folder, file)
        dst = os.path.join(os.path.join(output_folder, dst_folder), file)
        copyfile(src, dst)

    random.seed(12)
    train_docs = SameDocumentSampler(train_files, input_folder, storage_method, data_dir=output_folder)
    dev_docs = SameDocumentSampler(dev_files, input_folder, storage_method, data_dir=output_folder)
    test_docs = SameDocumentSampler(test_files, input_folder, storage_method, data_dir=output_folder)

    train_docs.generate_data(sample_number, "train_" + storage_method)
    dev_docs.generate_data(sample_number, "dev_" + storage_method)
    test_docs.generate_data(sample_number, "test_" + storage_method)
    print("Done!")
