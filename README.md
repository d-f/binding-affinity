```process_dataset.py``` processes the PDBBind dataset through the deepchem library. It will embed the sequence of amino acid abbreviations with a Protein LLM, convert the rdkit Molecule objects into torch_geometric graphs and saves the binding affinity each in separate directories. 
```determine_bond_types``` determines which bond types are present within the dataset for nomalization purposes
