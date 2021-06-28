import os
import json
import numpy as np

from tqdm import tqdm
from collections import Counter

if __name__ == "__main__":
    fp = "./tos-data-og"

    total_num_paras = []
    section_lengths = []
    number_of_times_section_topic_appeared = []
    paragraphs_of_section_topic = []

    for fn in tqdm(sorted(os.listdir(fp))):
        full_path = os.path.join(fp, fn)

        with open(full_path) as f:
            data = json.load(f)

        curr_num_paras = 1
        curr_section_length = 1
        prev_section_heading = data["level1_headings"][0]["section"]
        for section in data["level1_headings"][1:]:
            curr_num_paras += 1
            paragraphs_of_section_topic.append(section["section"])
            if section["section"] == prev_section_heading:
                curr_section_length += 1
            else:
                section_lengths.append(curr_section_length)
                curr_section_length = 1
                prev_section_heading = section["section"]
                number_of_times_section_topic_appeared.append(section["section"])

        total_num_paras.append(curr_num_paras)

    print(f"Median #Paras per Doc: {np.median(total_num_paras):.2f}")
    print(f"Mean #Paras per Doc:   {np.mean(total_num_paras):.2f}")
    print(f"---------------------------------------")
    print(f"Median #Paras per Section: {np.median(section_lengths):.2f}")
    print(f"Mean #Paras per Section:   {np.mean(section_lengths):.2f}")
    print("----------------------------------------")
    print("Top occurring sections by the number of times they appear:")
    print(Counter(number_of_times_section_topic_appeared).most_common(10))
    print("----------------------------------------")
    print("Top sections by the number of paragraphs:")
    print(Counter(paragraphs_of_section_topic).most_common(10))

