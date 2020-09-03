"""
GraphSeg crashes, but it doesn't sort files before working on them.
If files have too few paragaraphs it will cause an error.
Make sure we continue training only on the files that are not yet processed.
"""
import argparse

from tqdm import tqdm
import shutil
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='If graphseg crashes, make sure it resumes on the files '
                    'that were not parsed yet. ')
    parser.add_argument('--existing_dir',
                        help='Folder for all list of all files. ',
                        default="./graphseg-test")
    parser.add_argument('--new_dir',
                        help='New location to save the new files,'
                             'contains only those files that are yet to be processed.',
                        default="./graphseg-continue")
    parser.add_argument('--so_far_completed_dir',
                        help='The location of the folder with completed files.',
                        default="./../graphseg/og-test")

    args = parser.parse_args()
    existing_dir = args.existing_dir
    new_dir = args.new_dir
    so_far_completed_dir = args.so_far_completed_dir

    completed_files = set(os.listdir(so_far_completed_dir))

    # Complete deletion is necessary if it crashes multiple times
    shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(new_dir)

    for filename in tqdm(os.listdir(existing_dir)):
        src_fp = os.path.join(existing_dir, filename)
        target_fp = os.path.join(new_dir, filename)
        if filename not in completed_files:
            shutil.copyfile(src_fp, target_fp)
