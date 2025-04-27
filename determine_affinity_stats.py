import datasets
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import statistics
from typing import Tuple


def parse_cla():
    """
    parses command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_filepath", type=str)
    return parser.parse_args()


def determine_stats(dataset: datasets.arrow_dataset.Dataset) -> Tuple[float]:
    """
    determines mean and standard deviation of the negative log base 10 affinity values
    """
    values = []
    for ds_dict in tqdm(dataset):
        values.append(ds_dict["neg_log10_affinity_M"])
    avg_aff = sum(values) / len(values)
    std_aff = statistics.stdev(values)
    return avg_aff, std_aff


def save_results(file_path: str, avg_aff: float, std_aff: float) -> None:
    """
    saves the results in a JSON file
    """
    save_dict = {
        "avg_affinity": avg_aff,
        "std_affinity": std_aff
    }
    with open(file_path, mode="w") as fp:
        json.dump(save_dict, fp)


def main():
    args = parse_cla()
    dataset = load_dataset("jglaser/binding_affinity",split='train')
    avg_aff, std_aff = determine_stats(dataset)
    save_results(file_path=args.save_filepath, avg_aff=avg_aff, std_aff=std_aff)


if __name__ == "__main__":
    main()
