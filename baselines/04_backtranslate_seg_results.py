"""
"Translate" the results from Graphseg to a format that we can compare to.
We need to make sure the same file order as for the other files is kept here.
"""
import argparse
import csv
import json
import os

import numpy as np
from tqdm import tqdm


def get_labels(filename):
    with open(filename) as f:
        data = json.load(f)

    if len(data["level1_headings"]) < 2:
        return None

    labels = []

    last_header = data["level1_headings"][0]["section"]

    for para in data["level1_headings"][1:]:

        if para["section"] == last_header:
            labels.append(1)
        else:
            labels.append(0)
        last_header = para["section"]

    return labels


def get_random_baseline(labels):
    # Create random sampling baseline, knowing how many sections are in article
    curr_paragraphs = len(labels)
    curr_sections = len(labels) - sum(labels)
    rands = np.random.choice([0, 1],
                             len(labels),
                             replace=True,
                             p=[curr_sections / curr_paragraphs,
                                1 - (curr_sections / curr_paragraphs)])
    return rands


def get_textseg_result(filename, reference):
    with open(filename) as f:
        textseg_preds = f.readlines()

    with open(reference) as f:
        ref_paras = json.load(f)

    ref_paras = ref_paras["level1_headings"]

    prev_pred = None
    binary_preds = []
    # Last paragraph is always the same
    line_idx = 0
    for i, para in enumerate(ref_paras):
        para_text = para["text"].lower()
        recorded_break = False
        # iterate until the next segment is no longer in the paragraph.
        while True:
            if line_idx == len(textseg_preds):
                if i == len(ref_paras) - 1:
                    break  # We just had the last paragraph
                else:
                    # It's okay if the last line is just a single character.
                    if len(ref_paras[-1]["text"]) == 1 or \
                            len(set(ref_paras[-1]["text"].strip())) == 1:
                        break
                    raise ValueError(f" {i} {len(ref_paras)} Something went wrong with indexing in file {filename}")
            textseg_para = textseg_preds[line_idx].strip("\n").lower()
            if textseg_para in para_text:
                para_text = para_text.replace(textseg_para, "", 1)
                line_idx += 1
            elif textseg_para == "==========":
                para_text = para_text.replace(textseg_para, "", 1)
                line_idx += 1
                recorded_break = True
            else:
                break

        if recorded_break:
            binary_preds.append(0)
        else:
            binary_preds.append(1)

    return binary_preds


def get_graphseg_result(filename, reference):
    with open(filename) as f:
        graphseg_preds = f.readlines()

    with open(reference) as f:
        ref_paras = json.load(f)

    ref_paras = ref_paras["level1_headings"]

    prev_pred = None
    binary_preds = []
    # Last paragraph is always the same
    line_idx = 0
    for i, para in enumerate(ref_paras):
        para_text = para["text"]
        recorded_break = False
        # iterate until the next segment is no longer in the paragraph.
        while True:
            if line_idx == len(graphseg_preds):
                if i == len(ref_paras) - 1:
                    break  # We just had the last paragraph
                else:
                    if len(ref_paras[-1]["text"]) == 1:
                        break
                    raise ValueError(f" {i} {len(ref_paras)} Something went wrong with indexing")
            graphseg_para = graphseg_preds[line_idx].strip("\n")
            if graphseg_para in para_text:
                para_text = para_text.replace(graphseg_para, "", 1)
                line_idx += 1
            elif graphseg_para == "==========":
                para_text = para_text.replace(graphseg_para, "", 1)
                line_idx += 1
                recorded_break = True
            else:
                break

        if recorded_break:
            binary_preds.append(0)
        else:
            binary_preds.append(1)

    return binary_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and filter them '
                    'based on the selected topics, on section and paragraph levels.')
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the test data. ',
                        default="../og-test")
    parser.add_argument('--seg_folder',
                        help='The location for the output of textseg or '
                             'graphseg algorithm.',
                        default="../resources/graphseg/")
    parser.add_argument('--output_folder', help='The location for the output folder ',
                        default="../resources")
    parser.add_argument('--method', help='Either "textseg" or  "graphseg"',
                        choices=['textseg', 'graphseg'],
                        default="graphseg")
    args = parser.parse_args()

    regular_test_folder = args.input_folder
    textseg_folder = args.seg_folder
    evaluated_files = set(os.listdir(textseg_folder))
    preds = []
    for filename in tqdm(sorted(os.listdir(regular_test_folder))):
        fp = os.path.join(regular_test_folder, filename)
        labels = get_labels(fp)
        # Skip to have the same documents as other test sets.
        if not labels:
            continue
        # Some files weren't predicted, those are short ones.
        # Only load it if it actually has been processed.
        txt_name = filename[:-5] + ".txt"
        if txt_name in evaluated_files:
            if args.method == "textseg":
                votes = get_textseg_result(os.path.join(textseg_folder, txt_name),
                                           os.path.join(regular_test_folder, filename))
            else:
                votes = get_graphseg_result(os.path.join(textseg_folder, txt_name),
                                            os.path.join(regular_test_folder, filename))
            votes = votes[:-1]  # Everything until the last vote, due to setup
            if len(votes) != len(labels):
                print(f"\n{votes}")
                print(txt_name)
                print(len(votes), len(labels))
            preds.append(votes)
        # otherwise, we have to "improvise" by appending baseline
        else:
            print("Skipped file because not available")
            preds.append(labels)

    output_folder = args.output_folder
    output_name = "wiki_seg_results.csv" if args.method == "textseg" \
        else "graphseg_results.csv"
    csv_path = os.path.join(output_folder,output_name)
    with open(csv_path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for el in preds:
            writer.writerow(el)
    print(len(preds))
