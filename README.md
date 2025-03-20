```process_dataset.py``` processes the PDBBind dataset through the deepchem library. It will embed the sequence of amino acid abbreviations with a Protein LLM, convert the rdkit Molecule objects into torch_geometric graphs and saves the binding affinity each in separate directories. 

```determine_bond_types.py``` determines which bond types are present within the dataset for nomalization purposes

```train_models.py``` allows for training a pure GAT or a combination of a GAT and Transformer depending on what is set for the model_type parameter. If a pure GAT is used, protein LLM embeddings are concatenated to atomic features when creating a ligand molecule graph and the graph is used for whole graph regression. If a combination of transformer and GAT is used, the GAT will be used to embed the graph, and a transformer will predict the binding affinity between the protein embedding and embedded ligand graph.
