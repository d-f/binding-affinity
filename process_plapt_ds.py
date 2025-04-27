from tqdm import tqdm
import torch
from datasets import load_dataset
import argparse
import json
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Type


def parse_cla():
    """
    process command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-stat_json_path", type=str, default="C:\\personal_ML\\binding_affinity_prediction\\ds_stats.json")
    parser.add_argument("-save_dir", type=Path, default=Path("C:\\personal_ML\\binding_affinity_prediction\\processed\\"))
    return parser.parse_args()


def read_stats(stat_json_path: str) -> Dict:
    """
    read dataset statistics from JSON
    """
    with open(stat_json_path) as opened_json:
        return json.load(opened_json)


def norm_aff(aff_value: float, avg_aff: float, std_aff: float) -> float:
    """
    z-score normalization for affinity value
    """
    return (aff_value - avg_aff) / std_aff


def load_protbert(device: torch.device) -> Tuple[Type[BertTokenizer], Type[BertModel]]:
    """
    loads a pre-trained ProtBERT model
    """
    prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    prot_model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
    return prot_model, prot_tokenizer


def load_chemberta(device):
    """
    loads a pre-trained ChemBERTa model
    """
    mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mol_model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
    return mol_model, mol_tokenizer


class AffinityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ds_dict = self.dataset[idx]
        protein = ds_dict["seq"]
        ligand = ds_dict["smiles_can"]
        affinity = ds_dict["neg_log10_affinity_M"]

        return protein, ligand, affinity


def save_batch(file_counter, affinities, prots, ligs, save_dir):
    """
    saves a batch as .pt files
    """
    for i, (aff, prot, lig) in enumerate(zip(affinities, prots, ligs)):
        if not save_dir.joinpath(f"aff_{i+file_counter}.pt").exists():
            torch.save(aff, f=save_dir.joinpath(f"aff_{i+file_counter}.pt"))
        if not save_dir.joinpath(f"prot_{i+file_counter}.pt").exists():
            torch.save(prot, f=save_dir.joinpath(f"prot_{i+file_counter}.pt"))
        if not save_dir.joinpath(f"lig_{i+file_counter}.pt").exists():
            torch.save(lig, f=save_dir.joinpath(f"lig_{i+file_counter}.pt"))


def main():
    args = parse_cla()
    # read model statistics
    stat_dict = read_stats(stat_json_path=args.stat_json_path)

    # define torch device
    device = torch.device('cuda')

    # load embedding models
    prot_model, prot_tokenizer = load_protbert(device=device)
    lig_model, lig_tokenizer =  load_chemberta(device=device)

    # train gets the entire dataset, other subsets exclude various proteins
    entire_dataset = load_dataset("jglaser/binding_affinity", split="train[:10%]")

    # load DataLoader
    affinity_ds = AffinityDataset(dataset=entire_dataset)
    affinity_dl = DataLoader(affinity_ds, batch_size=20, shuffle=False)

    # keep track of a file_counter for file naming purposes
    file_counter = 0

    with torch.no_grad():
        for ds_tuple in tqdm(affinity_dl):
            # check if entire batch was finished, otherwise complete it
            if not args.save_dir.joinpath(f"lig_{len(ds_tuple[2])+file_counter}.pt").exists():  
                # normalize affinity values          
                normalized_affinities = [norm_aff(aff_value=x, avg_aff=stat_dict["avg_affinity"], std_aff=stat_dict["std_affinity"]) for x in ds_tuple[2]]
                
                # embed sequence of amino acids and delete input tokens to free memory
                tokenized_prots = prot_tokenizer(ds_tuple[0], padding=True, truncation=True, max_length=3200, return_tensors='pt')
                embedded_prots = prot_model(**tokenized_prots.to(device)).pooler_output
                del tokenized_prots
                
                # embed ligand and delete input tokens to free memory
                tokenized_ligs = lig_tokenizer(ds_tuple[1], padding=True, truncation=True, return_tensors='pt')
                embedded_ligs = lig_model(**tokenized_ligs.to(device)).pooler_output
                del tokenized_ligs

                # save the batch
                save_batch(file_counter=file_counter, affinities=normalized_affinities, prots=embedded_prots, ligs=embedded_ligs, save_dir=args.save_dir)
                # increment file coutner
                file_counter += len(normalized_affinities)

                # delete tensors to free memory
                del normalized_affinities
                del embedded_ligs
                del embedded_prots
                torch.cuda.empty_cache()
            else:
                # if file already exists, skip but still increment file_counter
                file_counter += len(ds_tuple[2])


if __name__ == "__main__":
    main()
