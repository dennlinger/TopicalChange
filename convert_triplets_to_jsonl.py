import os
import json
import regex
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", default="./train_reduction.tsv",
                        help="File to convert into .jsonl; has to be in triplet format already")
    parser.add_argument("output_file", default="./wiki727k.jsonl",
                        help="Name of the output file. Will create directory automatically")
    parser.add_argument("--min_length", type=int, default=50,
                        help="Will make sure that all entries in anchor/positive/negative have this as "
                             "their minimum character count.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    data = pd.read_csv(args.input_file, sep="\t", header=None)

    with open(args.output_file, "w") as f:
        for idx, row in data.iterrows():
            break_bit = False
            for col in range(3):
                # This catches cases where the Wikimedia identifier is in the triplet.
                if regex.search(r"\:[0-9]{4,10}$", row[col]):
                    break
                # Manually verify what happens to other lines
                elif len(row[col]) < args.min_length:
                    print(row[col])
            else:  # This means no faulty pairs have been found
                f.write(f"{json.dumps(list(row))}\n")


