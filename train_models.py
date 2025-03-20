from rdkit import Chem
import numpy as np
from pathlib import Path
from tqdm import tqdm
import Bio.PDB
from transformers import AutoTokenizer, AutoModel
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAT
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import argparse
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import r2_score
from typing import Type, Tuple
import torch.nn as nn


class AffinityGAT(torch.nn.Module):
    """
    GAT model that predicts affinity given a graph with protein embeddings
    as molecular features
    """
    def __init__(self, in_channels: int, gat_hidden: int, gat_drop: float, gat_layers: int) -> None:
        super(AffinityGAT, self).__init__()
        self.gat = GAT(
            in_channels=in_channels, 
            out_channels=1, 
            dropout=gat_drop, 
            hidden_channels=gat_hidden, 
            num_layers=gat_layers
            )

    def forward(
            self, 
            x: Type[torch.tensor], 
            edge_index: Type[torch.tensor], 
            batch: Type[torch.tensor]
            ) -> Type[torch.tensor]:
        x = self.gat(x, edge_index)
        x = global_mean_pool(x, batch)
        return x
    

class GATTransformer(torch.nn.Module):
    """
    Model that combines transformers with GAT modesl to separately embed
    protein amino acid sequences and ligand molecule graphs
    in order to predict binding affinity
    """
    def __init__(
            self, 
            in_channels: int, 
            gat_hidden: int, 
            embed_dim: int, 
            encoder_heads: int, 
            gat_drop: float, 
            gat_layers: int, 
            encoder_ff: int, 
            encoder_drop: float, 
            encoder_layers: int
            ) -> None:
        super(GATTransformer, self).__init__()
        self.gat = GAT(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            dropout=gat_drop, 
            hidden_channels=gat_hidden, 
            num_layers=gat_layers
            )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_ff,
            dropout=encoder_drop,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=encoder_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1),
        ) 

    def forward(
            self, 
            x: Type[torch.tensor], 
            edge_index: Type[torch.tensor], 
            prot_embed: Type[torch.tensor], 
            batch: Type[torch.tensor]
            ) -> Type[torch.tensor]:
        
        x = self.gat(x, edge_index)
        x = global_mean_pool(x, batch)
        stacked = torch.stack([prot_embed, x], dim=1)
        attended_vectors = self.transformer(stacked)
        regression_input = torch.mean(attended_vectors, dim=1)
        output = self.fc(regression_input)
        return output


class GATDataset(Dataset):
    """
    reads separate protein embedding, molecule graph and affinity files, 
    adds the protein embedding to the atomic features of the molecule graph
    """
    def __init__(self, save_dir: Path) -> None: 
        self.save_dir = save_dir
        self.protein_dir = self.save_dir.joinpath("protein")
        self.protein_files = [x.name for x in self.protein_dir.iterdir()]
        self.ligand_dir = self.save_dir.joinpath("ligand")
        self.ligand_files = [x.name for x in self.ligand_dir.iterdir()]
        self.affinity_dir = self.save_dir.joinpath("affinity")
        self.affinity_files = [x.name for x in self.affinity_dir.iterdir()]

        self.file_list = [x for x in self.protein_files if x in self.ligand_files and x in self.affinity_files]

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        file_name = self.file_list[idx]
        prot_embed = torch.tensor(torch.load(self.protein_dir.joinpath(file_name)))
        # weight_only=Fasle because data object contains custom classes
        lig_embed = torch.load(self.ligand_dir.joinpath(file_name), weights_only=False) 
        affinity = torch.load(self.affinity_dir.joinpath(file_name)).to(torch.float32)

        concat = np.zeros(shape=(lig_embed.x.shape[0], lig_embed.x.shape[1]+prot_embed.shape[0]))
        concat[:, :6] = lig_embed.x
        concat[:, 6:] = prot_embed
        lig_embed.x = torch.tensor(concat).to(torch.float32)
        return lig_embed, affinity
    

class TransformerDataset(Dataset):
    """
    reads protein embeddings, ligand graphs and affinity values
    separately and returns torch tensors of each
    """
    def __init__(self, save_dir: Path) -> None: 
        self.save_dir = save_dir
        self.protein_dir = self.save_dir.joinpath("protein")
        self.protein_files = [x.name for x in self.protein_dir.iterdir()]
        self.ligand_dir = self.save_dir.joinpath("ligand")
        self.ligand_files = [x.name for x in self.ligand_dir.iterdir()]
        self.affinity_dir = self.save_dir.joinpath("affinity")
        self.affinity_files = [x.name for x in self.affinity_dir.iterdir()]

        self.file_list = [x for x in self.protein_files if x in self.ligand_files and x in self.affinity_files]

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[Type[torch.tensor]]:
        file_name = self.file_list[idx]
        prot_embed = torch.tensor(torch.load(self.protein_dir.joinpath(file_name)))
        # weight_only=Fasle because data object contains custom classes
        lig_embed = torch.load(self.ligand_dir.joinpath(file_name), weights_only=False) 
        affinity = torch.load(self.affinity_dir.joinpath(file_name)).to(torch.float32)

        return prot_embed, lig_embed, affinity


def get_sequence_from_pdb(pdb_file: str) -> str:
    """
    reads a PDB file and returns a string of 
    amino acid abbreviations
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


def get_embedding(sequence: Type[torch.tensor], model_name: str) -> Type[np.array]:
    """
    takes a sequence of amino acid abbreviations and embeds them
    with a protein LLM
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    
    inputs = tokenizer(sequence, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state
    
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counts = torch.sum(attention_mask, 1, keepdim=True)
    mean_pooled = summed / counts
    
    return mean_pooled.numpy()[0]


def mol_to_graph(mol: Type[Chem.Mol]) -> Type[Data]:
    """
    converts a rdkit.Chem.Mol object into a 
    torch_geometric graph
    """
    atomic_features = []
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        degree = int(atom.GetDegree())
        implicit_valence = int(atom.GetImplicitValence())
        formal_charge = int(atom.GetFormalCharge())
        is_aromatic = 1 if atom.GetIsAromatic() else 0
        
        hybridization = int(atom.GetHybridization())
        chiral_tag = int(atom.GetChiralTag())
        
        features = [
            atomic_num,
            degree,
            implicit_valence,
            formal_charge,
            is_aromatic,
            hybridization,
            chiral_tag
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
            int(bond_type == 1), # single bond
            int(bond_type == 2), # double bond
            int(bond_type == 3), # triple bond
            int(bond_type == 12) # aromatic bond
        ]
        edge_features.append(features)
        edge_features.append(features)  # Add the same features for the reverse edge

    x = torch.tensor(atomic_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return data


def gat_dataloaders(save_dir: Path, batch_size: int) -> Tuple[Type[DataLoader]]:
    """
    loads torch geometric dataloaders
    """
    train_ds = GATDataset(save_dir=save_dir.joinpath("train"))
    val_ds = GATDataset(save_dir=save_dir.joinpath("val"))
    test_ds = GATDataset(save_dir=save_dir.joinpath("test"))

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl


def transformer_dataloaders(save_dir: Path, batch_size: int) -> Tuple[Type[DataLoader]]:
    """
    loads torch geometric dataloaders
    """
    train_ds = TransformerDataset(save_dir=save_dir.joinpath("train"))
    val_ds = TransformerDataset(save_dir=save_dir.joinpath("val"))
    test_ds = TransformerDataset(save_dir=save_dir.joinpath("test"))

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl


def eval_gat(
        model: Type[AutoModel], 
        val_dl: Type[DataLoader], 
        device: Type[torch.device], 
        loss_fn: Type[torch.nn.MSELoss]
        ) -> Tuple[float]:
    """
    evaluates a GAT model on a validation dataset and 
    returns the MSE loss and r^2 score
    """
    with torch.no_grad():
        model = model.eval()
        val_loss = 0
        ground_truth = []
        preds = []
        for lig, affinity in tqdm(val_dl):
            ground_truth += [num for num in affinity.numpy()]
            
            lig = lig.to(device)
            affinity = affinity.to(device)
            
            batch_idx = lig.batch

            out = model(x=lig.x, edge_index=lig.edge_index, batch=batch_idx)
            loss = loss_fn(out, affinity.unsqueeze(-1))

            val_loss += loss.item()

            preds += [num for num in out.cpu().detach().numpy()]

        val_loss /= len(val_dl)
        r2 = r2_score(y_true=ground_truth, y_pred=preds)

    return val_loss, r2

def eval_transformer(
        model: Type[AutoModel], 
        val_dl: Type[DataLoader], 
        device: Type[torch.device], 
        loss_fn: Type[torch.nn.MSELoss]
        ) -> Tuple[float]:
    """
    evaluates a Transformer model on a validation dataset and 
    returns the MSE loss and r^2 score
    """
    with torch.no_grad():
        model = model.eval()
        val_loss = 0
        ground_truth = []
        preds = []
        for prot, lig, affinity in tqdm(val_dl):
            ground_truth += [num for num in affinity.numpy()]
            
            prot = prot.to(device)
            lig = lig.to(device)
            affinity = affinity.to(device)
            
            batch_idx = lig.batch

            out = model(prot_embed=prot, x=lig.x, edge_index=lig.edge_index, batch=batch_idx)
            loss = loss_fn(out, affinity.unsqueeze(-1))

            val_loss += loss.item()

            preds += [num for num in out.cpu().detach().numpy()]

        val_loss /= len(val_dl)
        r2 = r2_score(y_true=ground_truth, y_pred=preds)

    return val_loss, r2


def train_gat(
        model: Type[AutoModel], 
        train_dl: Type[DataLoader], 
        val_dl: Type[DataLoader], 
        num_epochs: int, 
        device: Type[torch.device], 
        optim: Type[torch.optim.Adam], 
        loss_fn: Type[torch.nn.MSELoss], 
        patience: int,
        model_save_name: str, 
        save_dir: Path
        ) -> None:
    """
    trains a GAT model to predict binding affinity between
    a protein and ligand
    """
    model.train()
    patience_counter = 0
    lowest_loss = float("inf")
    for epoch_idx in range(num_epochs):
        if patience_counter == patience:
            break
        epoch_loss = 0
        for lig, affinity in tqdm(train_dl):
            lig = lig.to(device)
            affinity = affinity.to(device)
            
            batch_idx = lig.batch

            out = model(x=lig.x, edge_index=lig.edge_index, batch=batch_idx)
            loss = loss_fn(out, affinity.unsqueeze(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        mse, r2 = eval_gat(model=model, val_dl=val_dl, device=device, loss_fn=loss_fn)
        if mse < lowest_loss:
            lowest_loss = mse
            patience_counter = 0
            torch.save(model, f=save_dir.joinpath(model_save_name))
        else:
            patience_counter += 1
        print(f"Epoch {epoch_idx+1}:", epoch_loss)
        print(f"Validation: MSE {mse} R2 {r2}")


def train_transformer(
        model: Type[AutoModel], 
        train_dl: Type[DataLoader], 
        val_dl: Type[DataLoader], 
        num_epochs: int, 
        device: Type[torch.device], 
        optim: Type[torch.optim.Adam], 
        loss_fn: Type[torch.nn.MSELoss], 
        patience: int, 
        save_dir: Path, 
        model_save_name: str
        ) -> None:
    """
    trains a Transformer model to predict binding affinity between
    a protein and ligand
    """
    model.train()
    patience_counter = 0
    lowest_loss = float("inf")
    for epoch_idx in range(num_epochs):
        if patience_counter == patience:
            break
        epoch_loss = 0
        for prot, lig, affinity in tqdm(train_dl):
            prot = prot.to(device)
            lig = lig.to(device)
            affinity = affinity.to(device)
            
            batch_idx = lig.batch

            out = model(prot_embed=prot, x=lig.x, edge_index=lig.edge_index, batch=batch_idx)
            loss = loss_fn(out, affinity.unsqueeze(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        mse, r2 = eval_transformer(model=model, val_dl=val_dl, device=device, loss_fn=loss_fn)
        if mse < lowest_loss:
            lowest_loss = mse
            patience_counter = 0
            torch.save(model, f=save_dir.joinpath(model_save_name))
        else:
            patience_counter += 1
        print(f"Epoch {epoch_idx+1}:", epoch_loss)
        print(f"Validation: MSE {mse} R2 {r2}")


def parse_cla() -> None:
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    # directory where the saved, processed data are located
    parser.add_argument("-save_dir", type=Path)
    # number of instances to input into the model at once
    parser.add_argument("-batch_size", type=int)
    # proportion of the gradient to use for updating parameters
    parser.add_argument("-lr", type=float)
    # number of training iterations
    parser.add_argument("-num_epochs", type=int)
    # size of the GAT hidden state
    parser.add_argument("-gat_hidden", type=int) 
    # amount of dropout in the GAT model
    parser.add_argument("-gat_drop", type=float) 
    # number of layers in the GAT
    parser.add_argument("-gat_layers", type=int)
    # directory to save model
    parser.add_argument("-model_save_dir", type=Path)
    # filename of the model
    parser.add_argument("-model_save_name", type=str)
    # number of epochs past when the validation loss improves to continue training
    parser.add_argument("-patience", type=int)
    # determines whether a pure GAT is used or a combination of GAT and transformer
    parser.add_argument("-model_type", choices=["gat", "transformer"])
    # if model_type == transformer, number of heads in the transformer encoder
    parser.add_argument("-encoder_heads", type=int)
    # if model_type == transformer, number of transformer encoder layers
    parser.add_argument("-encoder_layers", type=int)
    # if model_type == transformer, size of the feed forward layer inside the encoder
    parser.add_argument("-encoder_ff", type=int)
    # if model_type == transformer, amount of dropout in the encoder
    parser.add_argument("-encoder_drop", type=float)
    return parser.parse_args()


def main():
    args = parse_cla()
    if args.model_type == "gat":
        model = AffinityGAT(
            in_channels=1286, 
            gat_hidden=args.gat_hidden, 
            gat_drop=args.gat_drop, 
            gat_layers=args.gat_layers
            )
        train_dl, val_dl, test_dl = gat_dataloaders(save_dir=args.save_dir, batch_size=args.batch_size)
    elif args.model_type == "transformer":
        model = GATTransformer(
        in_channels=6, 
        gat_hidden=args.gat_hidden, 
        embed_dim=1280, 
        encoder_heads=args.encoder_heads,
        gat_drop=args.gat_drop, 
        gat_layers=args.gat_layers, 
        encoder_ff=args.encoder_ff, 
        encoder_drop=args.encoder_drop, 
        encoder_layers=args.encoder_layers
        )
        train_dl, val_dl, test_dl = transformer_dataloaders(save_dir=args.save_dir, batch_size=args.batch_size)
    device = torch.device("cuda")
    model = model.to(device)
    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
    loss_fn = torch.nn.MSELoss()
    if args.model_type == "gat":
        train_gat(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            num_epochs=args.num_epochs,
            device=device,
            optim=optim,
            loss_fn=loss_fn,
            patience=args.patience, 
            model_save_name=args.model_save_name,
            save_dir=args.model_save_dir
        )
        model = torch.load(args.model_save_dir.joinpath(args.model_save_name), weights_only=False)
        test_mse, test_r2 = eval_gat(model=model, val_dl=test_dl, loss_fn=loss_fn, device=device)
        print("test MSE", test_mse)
        print("test r2", test_r2)

    elif args.model_type == "transformer":
        train_transformer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=args.num_epochs,
        device=device,
        optim=optim,
        loss_fn=loss_fn,
        patience=args.patience, 
        model_save_name=args.model_save_name,
        save_dir=args.model_save_dir
        )
        model = torch.load(args.model_save_dir.joinpath(args.model_save_name), weights_only=False)
        test_mse, test_r2 = eval_transformer(model=model, val_dl=test_dl, loss_fn=loss_fn, device=device)
        print("test MSE", test_mse)
        print("test r2", test_r2)


if __name__ == "__main__":
    main()
