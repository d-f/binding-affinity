```process_dataset.py``` processes the PDBBind dataset through the deepchem library. It will embed the sequence of amino acid abbreviations with a Protein LLM, convert the rdkit Molecule objects into torch_geometric graphs and saves the binding affinity each in separate directories. 

```determine_bond_types.py``` determines which bond types are present within the dataset for nomalization purposes

```train_affinity_gat.py``` the protein LLM embeddings are contatenated to each atomic feature for every atom in a ligand molecule graph and this script trains a GAT model to predict binding affinity.

