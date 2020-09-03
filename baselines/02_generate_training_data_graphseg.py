"""
Converts the document to a "paragraph-per-line" format for GraphSeq
"""
import argparse

from tqdm import tqdm
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and filter them based'
                    ' on the selected topics, on section and paragraph levels.')
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the raw data, '
                             'keep in mind the preprocessing steps. ',
                        default="./tos-data-og")
    parser.add_argument('--output_folder', help='The location for the output folder.',
                        default="../training_data_textseg")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(sorted(os.listdir(input_folder))):
        fp = os.path.join(input_folder, filename)

        with open(fp) as f:
            json_dict = json.load(f)

        graphseg_content = ""
        if len(json_dict["level1_headings"]) < 3:
            continue
        for para in json_dict["level1_headings"]:
            graphseg_content += para["text"]
            graphseg_content += "\n"

        filename = filename[:-5] + ".txt"
        out_fp = os.path.join(output_folder, filename)
        with open(out_fp, "w") as f:
            f.write(graphseg_content)
