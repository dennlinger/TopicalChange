"""
Merge the sections based on the key_dict that contains topics that should be merged and
create a new file for each of the ToS document based on the new section definition.
There is an option to decide whether to merge the paragraphs in the sections or not.
"""

import argparse
import json
import os

from utils import clean_title
from utils import key_dict, flip_dict, clean_text
from tqdm import tqdm


def extract_section(paragraph, temp_title, text, level, new_json):
    """
     Formulates the current section from paragraphs into a big text chunk
    :param paragraph: Current paragraph in the terms of service
    :param temp_title: The previous section's title. Equals None if the first section.
    :param text: Similar to temp_title, the previous section's text.
                 Will be appended if same section.
    :param level: Whether it is the first or second level heading.
    :param new_json: the new json file that we are creating with new headings
    :return:
    """
    title = clean_title(paragraph["section"][level], grouped_keys)
    if title:
        if temp_title is None:
            temp_title = title

        if temp_title == title:
            text = text + paragraph["text"] + "\n"
        else:
            new_json["level" + str(level + 1) + "_headings"].append({"section": temp_title,
                                                                     "text": text[:-2]})
            temp_title = title
            text = paragraph["text"] + "\n"

    return text, temp_title, new_json


def extract_section_paragraphs(paragraph, new_json, level=0):
    """
    Formulates the current section in to a cleaner representation with rectified labels.
    :param paragraph: Current paragraph in the terms of service
    :param new_json: The document where the new structure is stored for later output.
    :param level: Whether it is the first or second level heading.
    :return:
    """
    title = clean_title(paragraph["section"][level], grouped_keys)
    text = clean_text(paragraph["text"])
    if title and text:
        new_json["level" + str(level + 1) + "_headings"].append({"section": title,
                                                                 "text": text})

    return new_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Processing the raw terms of services dataset and filter them '
                    'based on the selected topics, on section and paragraph levels.')
    parser.add_argument('--paragraph', help='If set, will separate the paragraphs',
                        action='store_true', default=True)
    parser.add_argument('--input_folder',
                        help='The location of the folder containing the raw data.',
                        default="../resources/tos-data")
    parser.add_argument('--output_folder', help='The location for the output folder.',
                        default="../resources/tos-data-cleaned")

    args = parser.parse_args()

    grouped_keys = flip_dict(key_dict)

    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cleaned_counter = 0
    for f in tqdm(sorted(os.listdir(input_folder))):
        with open(os.path.join(input_folder, f)) as fp:
            tos = json.load(fp)
            new_json = {"level1_headings": [], "level2_headings": []}
            temp_title0 = None
            temp_title1 = None
            text0 = ""
            text1 = ""

            for paragraph in tos:
                # first level
                if len(paragraph["section"]) > 0:
                    text0, temp_title0, new_json = extract_section(paragraph,
                                                                   temp_title0,
                                                                   text0,
                                                                   0,
                                                                   new_json)
                    if args.paragraph:  # if it should be separated by paragraphs
                        new_json = extract_section_paragraphs(paragraph,
                                                              new_json,
                                                              level=0)

                    # second level
                    if len(paragraph["section"]) > 1:
                        text1, temp_title1, new_json = extract_section(paragraph,
                                                                       temp_title1,
                                                                       text1,
                                                                       1,
                                                                       new_json)
                        if args.paragraph:  # if it should be separated by paragraphs
                            new_json = extract_section_paragraphs(paragraph,
                                                                  new_json,
                                                                  level=1)

            # only save the file if some headings are matched
            if len(new_json["level1_headings"]) > 0:
                cleaned_counter = cleaned_counter + 1
                with open(os.path.join(output_folder, f), 'w') as f:
                    json.dump(new_json, f, ensure_ascii=False, indent=4)

    print("remaining cleaned files:", cleaned_counter)
