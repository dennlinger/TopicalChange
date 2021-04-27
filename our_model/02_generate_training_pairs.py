"""
Reads each section in each file and match with an other section from a different file
that has the same label, creates the same number of negative examples as well.
"""

from transformers import BertTokenizer, RobertaTokenizer
from sklearn.model_selection import train_test_split
from tokenizers import SentencePieceBPETokenizer
from tqdm import tqdm
import pandas as pd
import random
import json
import os
import argparse
import numpy as np
from shutil import copyfile


class Doc(object):
    def __init__(self, storage_method, force_shorten, data_dir, tokenizer_path):
        self.all_lens = {}
        self.num_labels = None
        self.final_data = []
        self.final_triplet_data = []
        self.tokenizer_path = tokenizer_path

        self.storage_method = storage_method
        self.force_shorten = force_shorten
        self.tokenizer = None
        self.set_tokenizer()

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

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
        else:  # the case for our own embedding
            encoded_text = self.tokenizer.encode(text).ids
            if self.force_shorten and len(encoded_text) > 512:
                return encoded_text[:512]  # Own encoding has no special symbols
            else:
                return encoded_text

    def _add_samples_triplet_loss(self, section_pos, section_neg, section):
        self.final_triplet_data.append({"section_center": section,
                                        "section_pos": section_pos,
                                        "section_neg": section_neg})

    def generate_positive_samples(self, label, section, doc):
        second_section = self._add_samples(label, section, doc, 1)
        return second_section

    def generate_negative_samples(self, label, section, doc):
        # choose a random label as negative
        rand_neg_label = label
        while rand_neg_label == label:
            rand_neg_label = random.choice(list(doc.keys()))
        second_section = self._add_samples(rand_neg_label, section, doc, 0)
        return second_section


class Documents(object):
    def __init__(self, inputs, labels, storage_method="raw", force_shorten=True,
                 data_dir="./data", tokenizer_path="./"):
        """
        Essentially bucket-sort all inputs according to their labels.
        Then we can simply "draw" from the correct bucket
        :param inputs: str, Input texts
        :param labels: str, Corresponding labels
        :param storage_method: Either "raw" (remove \n and \t),
                               "bert" (BERT tokenization),
                               "roberta" (RoBERTa tokenization), or
                               "token" (own tokenizer)
        :param force_shorten: Force a shortening for BERT-based models to fit.
        :param data_dir: where to store the output
        :param tokenizer_path: where to find the sentence piece tokenizer

        """
        Doc.__init__(self, storage_method, force_shorten, data_dir, tokenizer_path)
        self.all_docs = {}

        for text, label in zip(inputs, labels):
            self.add_to_section(text, label)
        self.compute_lens()

        self.all_labels = list(self.all_docs.keys())

    def _add_samples(self, label, section, value):
        rand_pos = random.randint(0, self.all_lens[label] - 1)
        second_section = self.all_docs[label][rand_pos]
        self.final_data.append({"section_1": section, "section_2": second_section, "label": value})
        return second_section

    def add_to_section(self, section_text, section_label):
        processed_text = self.transform_text_accordingly(section_text)

        if section_label in self.all_docs:
            self.all_docs[section_label].append(processed_text)
        else:
            self.all_docs[section_label] = [processed_text]

    def compute_lens(self):
        # keep all the lens for random generation later on
        for label, text in self.all_docs.items():
            self.all_lens[label] = len(self.all_docs[label])
        self.num_labels = len(self.all_docs)

    def generate_data(self, sample_number, file_name):
        """
        :param sample_number: Number of positive/negative samples, respectively.
        :param file_name: Storage file name (train, test, dev)
        :return: None
        """
        print(f"Generating {file_name} data...")
        for label, sections in tqdm(self.all_docs.items()):
            for section in sections:
                for i in range(sample_number):
                    section_pos = self.generate_positive_samples(label, section)

                    section_neg = self.generate_negative_samples(label, section)

                    self._add_samples_triplet_loss(section, section_pos, section_neg)

        print(f"{len(self.final_data)} sample generated.")
        print("writing data to tsv file...")
        df_data = pd.DataFrame(self.final_data, columns=["section_1", "section_2", "label"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '.tsv'), sep='\t', index=False, header=False)
        df_data = pd.DataFrame(self.final_triplet_data, columns=["section_center", "section_pos", "section_neg"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '_triplet.tsv'), sep='\t', index=False, header=False)


class ConsecutiveDocuments(Doc):
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
        Doc.__init__(self, storage_method, force_shorten, data_dir, tokenizer_path)
        self.all_docs = []

        for f in tqdm(files):
            doc = {}
            with open(os.path.join(folder, f)) as fp:
                tos = json.load(fp)
            for section in tos:
                # Transform dict into X/y sample
                text = section["Text"]
                label = section["Section"]
                doc = self.add_to_section(text, label, doc)

            self.all_docs.append(doc)

    def add_to_section(self, section_text, section_label, doc):
        processed_text = self.transform_text_accordingly(section_text)

        if section_label in doc:
            doc[section_label].append(processed_text)
        else:
            doc[section_label] = [processed_text]
        return doc

    def _add_samples(self, label, section, doc, value):
        rand_pos = random.randint(0, len(doc[label]) - 1)
        second_section = doc[label][rand_pos]
        self.final_data.append({"section_1": section,
                                "section_2": second_section,
                                "label": value})
        return second_section

    def generate_data(self, sample_number, file_name):
        """
        :param sample_number: Number of positive/negative samples, respectively.
        :param file_name: Storage file name (train, test, dev)
        :return: None
        """
        docs_skipped = 0
        sections_skipped = 0
        print(f"Generating {file_name} data...")
        for doc in tqdm(self.all_docs):
            for label, sections in doc.items():
                if len(doc) < 2:
                    docs_skipped += 1
                    continue
                for section in sections:
                    if len(sections) >= sample_number:
                        for i in range(sample_number):
                            section_pos = self.generate_positive_samples(label, section, doc)

                            section_neg = self.generate_negative_samples(label, section, doc)

                            self._add_samples_triplet_loss(section, section_pos, section_neg)
                    else:
                        sections_skipped += 1
                        continue

        print(f"{len(self.final_data)} sample generated. "
              f"{docs_skipped} docs skipped, {sections_skipped} sections skipped! ")
        print("writing data to tsv file...")
        df_data = pd.DataFrame(self.final_data, columns=["section_1", "section_2", "label"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '_consecutive.tsv'),
                       sep='\t', index=False, header=False)
        df_data = pd.DataFrame(self.final_triplet_data,
                               columns=["section_center", "section_pos", "section_neg"])
        df_data.to_csv(os.path.join(self.data_dir, file_name + '_consecutive_triplet.tsv'),
                       sep='\t', index=False, header=False)


def load_files(files, folder):
    inputs = []
    labels = []
    for f in tqdm(files):
        with open(os.path.join(folder, f)) as fp:
            tos = json.load(fp)

        for section in tos:
            # Transform dict into X/y sample
            text = section["Text"]
            label = section["Section"]

            inputs.append(text)
            labels.append(label)

    return inputs, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and filter them '
                    'based on the selected topics, on section and paragraph levels. '
                    'To generate training data for Same topic predictation tasks" ')
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the raw data, '
                             'keep in mind the preprocessing steps.',
                        default="../resources/tos-data-og")
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="../resources/training_data")
    parser.add_argument('--sentence_tokenizer_path',
                        help='Path to the sentence tokenizer folder, '
                             'set this if you are using the storage_method=token.',
                        default="./")
    parser.add_argument('--sample_number',
                        help='Number of positive and negative samples to be created '
                             'from each section or paragraph', type=int, default=3)
    parser.add_argument('--heading_level', help='first or second level headings',
                        type=int, default=1)
    parser.add_argument('--storage_method',
                        help='Either "raw",  "bert" (BERT tokenization), '
                             '"roberta" (RoBERTa tokenization), or "token" (own tokenizer)',
                        choices=['raw', 'roberta', 'bert', 'token'], default="raw")
    parser.add_argument('--random_split',
                        help='If set to true all data will be read and combined before spliting, '
                             'otherwise the documents are separated for dev and test',
                        action='store_true', default=False)
    parser.add_argument('--consecutive',
                        help='If set then the topic structure is ignored and '
                             'the positive samples come from the pararaphs that '
                             'belong to the same section in the same document. '
                             'Does not work at the same time as random_split.',
                        action='store_true', default=True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    sample_number = args.sample_number  # number of positive and negative samples to be create from each section
    heading_level = args.heading_level  # first or second level headings
    storage_method = args.storage_method  # Which method to store?
    tokenizer_path = args.sentence_tokenizer_path

    all_inputs = []
    all_labels = []

    if args.random_split:
        for f in tqdm(sorted(os.listdir(input_folder))):
            with open(os.path.join(input_folder, f)) as fp:
                tos = json.load(fp)

            for section in tos["level" + str(heading_level) + "_headings"]:
                # Transform dict into X/y sample
                text = section["text"]
                label = section["section"]

                all_inputs.append(text)
                all_labels.append(label)

        # use 20% for remainder and then split again for 10% of total for dev/test each
        input_train, input_rest, label_train, label_rest = train_test_split(all_inputs, all_labels, test_size=0.2,
                                                                            random_state=69120, stratify=all_labels)

        input_dev, input_test, label_dev, label_test = train_test_split(input_rest, label_rest, test_size=0.5,
                                                                        random_state=1312, stratify=label_rest)
    else:
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
        input_test, label_test = load_files(test_files, input_folder)
        input_dev, label_dev = load_files(dev_files, input_folder)

        dst_folder = "og-test"  # copy the files from the test set to the new location
        # for text segmentation we need the original structure of the files in the test set.
        os.makedirs(os.path.join(output_folder, dst_folder), exist_ok=True)
        for file in test_files:
            src = os.path.join(input_folder, file)
            dst = os.path.join(os.path.join(output_folder, dst_folder), file)
            copyfile(src, dst)

    random.seed(12)
    if args.consecutive:

        train_docs = ConsecutiveDocuments(train_files, input_folder, storage_method, data_dir=output_folder)
        test_docs = ConsecutiveDocuments(dev_files, input_folder, storage_method, data_dir=output_folder)
        dev_docs = ConsecutiveDocuments(test_files, input_folder, storage_method, data_dir=output_folder)
    else:
        train_docs = Documents(input_train, label_train, storage_method, data_dir=output_folder)
        test_docs = Documents(input_test, label_test, storage_method, data_dir=output_folder)
        dev_docs = Documents(input_dev, label_dev, storage_method, data_dir=output_folder)

    train_docs.generate_data(sample_number, "train_" + storage_method)
    test_docs.generate_data(sample_number, "test_" + storage_method)
    dev_docs.generate_data(sample_number, "dev_" + storage_method)
    print("Done!")
