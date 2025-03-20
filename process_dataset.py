import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import deepchem as dc
import Bio.PDB
from transformers import AutoTokenizer, AutoModel
import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import Type, Dict, Tuple


def get_sequence_from_pdb(pdb_file: str) -> str:
    """
    reads a PDB file and returns a string of amino acid abbreviations
    """
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if Bio.PDB.Polypeptide.is_aa(residue.get_resname(), standard=True):
                    sequence += Bio.PDB.Polypeptide.index_to_one(Bio.PDB.Polypeptide.three_to_index(residue.get_resname()))
    
    return sequence


def get_embedding(sequence: torch.tensor, tokenizer: Type[AutoTokenizer], model: Type[AutoModel]) -> np.array:
    """
    embeds the sqeuence of amino acid abbreviations
    with a protein LLM
    """
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", padding=True)
        outputs = model(**inputs)
    
        embeddings = outputs.last_hidden_state
        
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.sum(attention_mask, 1, keepdim=True)
        mean_pooled = summed / counts
        
        return mean_pooled.numpy()[0]


def mol_to_graph(mol: Type[Chem.Mol], max_mol: Dict, min_mol: Dict) -> Type[Data]:
    """
    converts a Chem.Mol object into a torch_geometric.data.Data object
    """
    atomic_features = []
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        degree = int(atom.GetDegree())
        formal_charge = int(atom.GetFormalCharge())
        is_aromatic = 1 if atom.GetIsAromatic() else 0
        
        hybridization = int(atom.GetHybridization())
        chiral_tag = int(atom.GetChiralTag())
        
        features = [
            (atomic_num - min_mol["atomic_num"]) / (max_mol["atomic_num"] - min_mol["atomic_num"]),
            (degree - min_mol["degree"]) / (max_mol["degree"] - min_mol["degree"]),
            (formal_charge - min_mol["formal_charge"]) / (max_mol["formal_charge"] - min_mol["formal_charge"]),
            is_aromatic,
            (hybridization - min_mol["hybridization"]) / (max_mol["hybridization"] - min_mol["hybridization"]),
            (chiral_tag - min_mol["chiral_tag"]) / (max_mol["chiral_tag"] - min_mol["chiral_tag"]),
        ]
        atomic_features.append(features)

    edge_indices = [[], []]
    edge_features = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices[0].append(start)
        edge_indices[0].append(end)

        edge_indices[1].append(start)
        edge_indices[1].append(end)

        bond_type = bond.GetBondType().real
        features = [
            int(bond_type == 1) / 12, # single bond
            int(bond_type == 2) / 12, # double bond
            int(bond_type == 3) / 12, # triple bond
            int(bond_type == 12) / 12 # aromatic bond
        ]
        edge_features.append(features)
        edge_features.append(features)  # Add the same features for the reverse edge

    x = torch.tensor(atomic_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return data


def affinity_min_max(
        train_dataset: Type[dc.data.DiskDataset], 
        val_dataset: Type[dc.data.DiskDataset], 
        test_dataset: Type[dc.data.DiskDataset]
        ) -> Tuple[Dict]:
    """
    retrieves the maximum and minumum affinity values for normalization
    """
    max_affinity = 0
    min_affinity = float("inf")

    for affinity in train_dataset.y:
        max_affinity = max(max_affinity, affinity)    
        min_affinity = min(min_affinity, affinity)    

    for affinity in val_dataset.y:
        max_affinity = max(max_affinity, affinity)    
        min_affinity = min(min_affinity, affinity)    

    for affinity in test_dataset.y:
        max_affinity = max(max_affinity, affinity)    
        min_affinity = min(min_affinity, affinity)    

    return max_affinity, min_affinity


def save_ds(
        dataset: Type[dc.data.DiskDataset], 
        save_dir: Path, 
        max_affinity: np.float, 
        min_affinity: np.float, 
        max_mol: Dict, 
        min_mol: Dict, 
        model: Type[AutoModel], 
        tokenizer: Type[AutoTokenizer]
        ) -> None:
    """
    saves and processess a DeepChem DiskDataset for PDBBind 
    """
    if not save_dir.joinpath("protein").exists():
        save_dir.joinpath("protein").mkdir(parents=False, exist_ok=True)
    if not save_dir.joinpath("ligand").exists():
        save_dir.joinpath("ligand").mkdir(parents=False, exist_ok=True)
    if not save_dir.joinpath("affinity").exists():
        save_dir.joinpath("affinity").mkdir(parents=False, exist_ok=True)

    for i, ds_tuple in enumerate(tqdm(dataset.X)):
        try:
            file_name_id = Path(ds_tuple[0]).name.partition("_ligand.sdf")[0]

            if not save_dir.joinpath("protein").joinpath(f"{file_name_id}.pt").exists():
                aminos = get_sequence_from_pdb(ds_tuple[1])
                prot_embed = torch.tensor(get_embedding(sequence=aminos, tokenizer=tokenizer, model=model))
                torch.save(prot_embed, str(save_dir.joinpath("protein").joinpath(f"{file_name_id}.pt")))

            if not save_dir.joinpath("ligand").joinpath(f"{file_name_id}.pt").exists():
                supplier = Chem.SDMolSupplier(ds_tuple[0])
                molecules = [mol for mol in supplier]
                if molecules[0]:
                    mol_graph = mol_to_graph(molecules[0], max_mol, min_mol)
                    torch.save(mol_graph, str(save_dir.joinpath("ligand").joinpath(f"{file_name_id}.pt")))

            if not save_dir.joinpath("affinity").joinpath(f"{file_name_id}.pt").exists():
                affinity = torch.tensor((dataset.y[i] - min_affinity) / (max_affinity - min_affinity)) 
                torch.save(affinity, str(save_dir.joinpath("affinity").joinpath(f"{file_name_id}.pt")))
        except:
            print("skipped")


def mol_feat_min_max(
        train_dataset: Type[dc.data.DiskDataset], 
        val_dataset: Type[dc.data.DiskDataset], 
        test_dataset: Type[dc.data.DiskDataset]
        ) -> Tuple[Dict]:
    """
    gets the maximum and minimum of each of the molecular features for normalization
    """
    max_dict = {
        "atomic_num": 0,
        "degree": 0,
        "formal_charge": 0,
        "hybridization": 0,
        "chiral_tag": 0,
    }

    min_dict = {
        "atomic_num": float("inf"),
        "degree": float("inf"),
        "formal_charge": float("inf"),
        "hybridization": float("inf"),
        "chiral_tag": float("inf"),
    }

    for dataset in [train_dataset, val_dataset, test_dataset]:
        for ds_tuple in dataset.X:
            try:
                supplier = Chem.SDMolSupplier(ds_tuple[0])
                molecule = [mol for mol in supplier][0]
                for atom in molecule.GetAtoms():
                    atomic_num = int(atom.GetAtomicNum())
                    degree = int(atom.GetDegree())
                    formal_charge = int(atom.GetFormalCharge())
                    
                    hybridization = int(atom.GetHybridization())
                    chiral_tag = int(atom.GetChiralTag())
                    
                    max_dict["atomic_num"] = max(atomic_num, max_dict["atomic_num"])
                    max_dict["degree"] = max(degree, max_dict["degree"])
                    max_dict["formal_charge"] = max(formal_charge, max_dict["formal_charge"])
                    max_dict["hybridization"] = max(hybridization, max_dict["hybridization"])
                    max_dict["chiral_tag"] = max(chiral_tag, max_dict["chiral_tag"])

                    min_dict["atomic_num"] = min(atomic_num, max_dict["atomic_num"])
                    min_dict["degree"] = min(degree, max_dict["degree"])
                    min_dict["formal_charge"] = min(formal_charge, max_dict["formal_charge"])
                    min_dict["hybridization"] = min(hybridization, max_dict["hybridization"])
                    min_dict["chiral_tag"] = min(chiral_tag, max_dict["chiral_tag"])
            except:
                print("skipped")

    return max_dict, min_dict


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_dir", type=str)
    parser.add_argument("-dataset_dir", type=str)
    parser.add_argument("-train_dir", type=Path)
    parser.add_argument("-val_dir", type=Path)
    parser.add_argument("-test_dir", type=Path)

    return parser.parse_args()

 
def main():
    args = parse_cla()
    _, datasets, _, = dc.molnet.load_pdbbind(
        featurizer="raw",
        set_name="general",
        save_dir=args.save_dir,
        data_dir=args.dataset_dir,
        ignore_missing_files=True
    )

    max_aff, min_aff = affinity_min_max(train_dataset=datasets[0], val_dataset=datasets[1], test_dataset=datasets[2])
    max_mol, min_mol = mol_feat_min_max(train_dataset=datasets[0], val_dataset=datasets[1], test_dataset=datasets[2])
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    model.eval()

    save_ds(
        dataset=datasets[0], 
        save_dir=args.train_dir,
        max_affinity=max_aff, 
        min_affinity=min_aff,
        max_mol=max_mol,
        min_mol=min_mol,
        model=model,
        tokenizer=tokenizer
        )
    save_ds(
        dataset=datasets[1], 
        save_dir=args.val_dir,
        max_affinity=max_aff, 
        min_affinity=min_aff,
        max_mol=max_mol,
        min_mol=min_mol,
        model=model,
        tokenizer=tokenizer
        )
    save_ds(
        dataset=datasets[2], 
        save_dir=args.test_dir,
        max_affinity=max_aff, 
        min_affinity=min_aff,
        max_mol=max_mol,
        min_mol=min_mol,
        model=model,
        tokenizer=tokenizer
        )


if __name__ == "__main__":
    main()
