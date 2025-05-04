import csv
import random
from pathlib import Path
import argparse
from typing import List, Tuple


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_folder", 
        type=Path
        )
    parser.add_argument(
        "-save_folder", 
        type=Path
        )
    parser.add_argument("-eval_prop", type=float, default=0.1)
    return parser.parse_args()


def filter_files(pattern: str, folder: Path) -> List:
    """
    filters files that contain pattern
    """
    filtered = folder.glob(f"*{pattern}*")
    return [x.name for x in filtered]


def partition_ds(aff_files: List, eval_prop: float) -> Tuple[List]:
    """
    partitions dataset into training,
    validation and test files
    """
    eval_amt = int(eval_prop * len(aff_files))
    val = random.sample(aff_files, eval_amt)
    aff_files = [x for x in aff_files if x not in val]

    test = random.sample(aff_files, eval_amt)
    aff_files = [x for x in aff_files if x not in test]

    return aff_files, val, test


def save_ds(save_folder: Path, train: List, val: List, test: List) -> None:
    """
    saves the dataset lists in CSV files
    """
    train_list = []
    val_list = []
    test_list = []

    for aff_file in train:
        idx = aff_file.partition("_")[2].partition(".pt")[0]
        train_list.append((aff_file, f"prot_{idx}.pt", f"lig_{idx}.pt"))
    for aff_file in val:
        idx = aff_file.partition("_")[2].partition(".pt")[0]
        val_list.append((aff_file, f"prot_{idx}.pt", f"lig_{idx}.pt"))
    for aff_file in test:
        idx = aff_file.partition("_")[2].partition(".pt")[0]
        test_list.append((aff_file, f"prot_{idx}.pt", f"lig_{idx}.pt"))
    
    with open(save_folder.joinpath("train.csv"), mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in train_list:
            writer.writerow(row)
    with open(save_folder.joinpath("val.csv"), mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in val_list:
            writer.writerow(row)
    with open(save_folder.joinpath("test.csv"), mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in test_list:
            writer.writerow(row)


def main():
    args = parse_cla()
    aff_files = filter_files(pattern="aff_", folder=args.data_folder)
    train, val, test = partition_ds(aff_files=aff_files, eval_prop=args.eval_prop)
    save_ds(save_folder=args.save_folder, train=train, val=val, test=test)


if __name__ == "__main__":
    main()
