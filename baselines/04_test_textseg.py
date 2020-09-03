import argparse
import json
import os
from collections import OrderedDict

import gensim
import numpy as np
import torch
from textseg import utils, text_manipulation
from textseg.utils import maybe_cuda
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
from textseg import wiki_utils

preds_stats = utils.predictions_analysis()

# This code has to be run the textseg enviroment in python 2.
# Loads the already trained model form the textseg
# and does textsegemention on a per document bases, placing
# "==========" where there is a new section.


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()


class LastUpdatedOrderedDict(OrderedDict):
    """
    Store items in the order the keys were last added.
    We need this to keep the structure of the file.
    """

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


class TextSegDataGenerator(object):
    def __init__(self, files, input_folder):
        """
        Generates input data in the format of the textseg model, but per document basis
        :param files: files to process
        :param input_folder: the folder containing all the input files
        """
        self.all_docs = []
        self.docs_names =[]

        # Create data directory

        for f in tqdm(sorted(files)):
            doc = LastUpdatedOrderedDict()
            with open(os.path.join(input_folder, f)) as fp:
                tos = json.load(fp)
            # Now equal to the test set?
            if len(tos["level1_headings"]) < 2 :
                continue
            for section in tos["level1_headings"]:
                # Transform dict into X/y sample
                text = section["text"].lower()
                label = section["section"]
                doc = self.add_to_section(text, label, doc)

            self.all_docs.append(doc)
            self.docs_names.append(f)

    def add_to_section(self, section_text, section_label, doc):
        processed_text = section_text

        if section_label in doc:
            doc[section_label]+= processed_text+"\n"
        else:
            doc[section_label] = processed_text+"\n"
        return doc

    @staticmethod
    def count_str_occurrences(string, findStr):
        return len(string.split(findStr)) - 1

    def process_section(self, section, id):
        global num_sentences_for_avg
        global sum_sentences_for_avg
        num_sentences_for_avg = 0
        sum_sentences_for_avg = 0
        sentences = text_manipulation.split_sentences(section, id)
        section_sentences = []
        num_sentences = 0
        num_formulas = 0
        num_codes = 0
        for sentence in sentences:
            sentence_words = text_manipulation.extract_sentence_words(sentence)

            sum_sentences_for_avg += len(sentence_words)
            num_sentences_for_avg += 1

            num_formulas += self.count_str_occurrences(sentence,
                                                       wiki_utils.get_formula_token())
            num_codes += self.count_str_occurrences(sentence,
                                                    wiki_utils.get_codesnipet_token())
            num_sentences += 1
            section_sentences.append(sentence)
        return section_sentences

    def generate_data(self, word2vec):
        """
        :param word2vec: the word2vec model for words
        :return: [data]: lis of all the transformed sentences ,
                 [targets]: list of number of sentences in each section ,
                 section_sentences_out: list of sentences in that section,
                 name: name of the file
        """

        for doc, name in zip(self.all_docs, self.docs_names):
            targets = []
            data = []
            section_sentences_out = []
            for(label, sections) in doc.items():
                section_sentences = self.process_section(sections, label)

                for sentence in section_sentences:
                    sentence_words = text_manipulation.\
                        extract_sentence_words(sentence, remove_special_tokens=False)
                    if 1 <= len(sentence_words):
                        data.append(maybe_cuda(
                            torch.tensor([text_manipulation.word_model(word, word2vec)
                                          for word in sentence_words]).reshape(len(sentence_words), 300)))
                        section_sentences_out.append(sentence)
                    else:
                        print('Sentence in wikipedia file is empty!')
                if data:
                    targets.append(len(data) - 1)

            yield [data], [targets], section_sentences_out, name


def test(model, threshold, test_data, files, output_folder):
    """
    generates segemenation based on the pretraind model
    :param model: textseg model
    :param threshold: the threshold for which the output is considered  as section break
    :param test_data: the generator object
    :param files: the list of all files
    :param output_folder: where to store the files
    :return:
    """
    model.eval()
    with tqdm(desc='Testing', total=len(files)) as pbar:
        for data, target, section_sentences_out, name in test_data:
            pbar.update()
            output = model(data)
            output_prob = softmax(output.data.cpu().numpy())
            output = output_prob[:, 1] > threshold

            with open(os.path.join(output_folder, name.replace("json","txt")),'wb') as fp:
                counter=0
                for sen in section_sentences_out:
                    if (len(section_sentences_out)-1) == len(output):
                        if counter < (len(section_sentences_out) - 1):
                            pred = output[counter]
                        else:
                            pred = True

                    fp.write((sen+"\n").encode('utf-8'))
                    if pred:
                        fp.write(("=========="+"\n").encode('utf-8'))
                    counter = counter+1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and filter them '
                    'based on the selected topics, on section and paragraph levels.')
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the test data. ',
                        default="../resources/og-test")
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="../resources/text_seg_output")
    parser.add_argument('--path_to_model',
                        help='The location for the trained textseg model. ',
                        default="../resources/textseg_model/best_model.t7")
    parser.add_argument('--path_to_word2vec_model',
                        help='The location for the word2vec model form textseg. ',
                        default="../resources/word2vec")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(args.path_to_word2vec_model,
                                                               binary=True)

    with open(args.path_to_model, 'rb') as f:
        model = torch.load(f)
    model = maybe_cuda(model)

    files = sorted(os.listdir(input_folder))
    test_docs = TextSegDataGenerator(files,input_folder=input_folder)

    test(model, 0.4, test_docs.generate_data(word2vec), files, output_folder)



