import deepchem as dc
from rdkit import Chem
from tqdm import tqdm
import argparse
from typing import Type


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_dir", type=str)
    parser.add_argument("-dataset_dir", type=str)
    return parser.parse_args()


def get_bonds(mol: Type[Chem.Mol]) -> set:
    """
    returns a set of the different bond types found
    within the dataset
    """
    bond_types = set()

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType().real
        bond_types.add(bond_type)

    return bond_types


def main():
    args = parse_cla()
    _, datasets, _, = dc.molnet.load_pdbbind(
    featurizer="raw",
    split="random",
    subset="refined",
    save_dir=args.save_dir,
    data_dir=args.dataset_dir    
    )
    bond_types = set()

    for dataset in datasets:
        for data_tuple in tqdm(dataset.X):
            supplier = Chem.SDMolSupplier(data_tuple[0])
            molecules = [mol for mol in supplier if mol is not None]
            for molecule in molecules:
                bond_types.update(get_bonds(molecule))

    print(bond_types)


if __name__ == "__main__":
    main()
