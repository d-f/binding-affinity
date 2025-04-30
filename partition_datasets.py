import json
from tqdm import tqdm
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset
import csv
import argparse
import torch
from pathlib import Path
from typing import List, Tuple, Type


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_folder", type=Path)
    parser.add_argument("-csv_folder", type=Path)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-num_epochs", type=int, default=32)
    return parser.parse_args()


def read_ds_csvs(csv_folder):
    """
    reads dataset CSV files
    """
    with open(csv_folder.joinpath("train.csv")) as opened_csv:
        reader = csv.reader(opened_csv)
        train = [x for x in reader]

    with open(csv_folder.joinpath("val.csv")) as opened_csv:
        reader = csv.reader(opened_csv)
        val = [x for x in reader]

    with open(csv_folder.joinpath("test.csv")) as opened_csv:
        reader = csv.reader(opened_csv)
        test = [x for x in reader]
    return train, val, test
        


class AffinityDataset(Dataset):
    """
    dataset to return the affinity, protein and ligand 
    files given a partitioned list of affinity file names
    """
    def __init__(self, data_folder: Path, dataset_list: List) -> None:
        self.data_folder = data_folder
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx: int) -> Tuple[Type[torch.tensor]]:
        aff_file, prot_file, lig_file = self.dataset_list[idx]
        aff = torch.load(self.data_folder.joinpath(aff_file))
        prot = torch.load(self.data_folder.joinpath(prot_file))
        lig = torch.load(self.data_folder.joinpath(lig_file))

        return prot, lig, aff


def create_dataloaders(csv_folder: Path, data_folder: Path, batch_size: int) -> Tuple[Type[DataLoader]]:
    """
    creates the dataloaders for the different
    dataset partitions
    """
    train, val, test = read_ds_csvs(csv_folder=csv_folder)
    train_ds = AffinityDataset(data_folder=data_folder, dataset_list=train)
    val_ds = AffinityDataset(data_folder=data_folder, dataset_list=val)
    test_ds = AffinityDataset(data_folder=data_folder, dataset_list=test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


class PLAPT(torch.nn.Module):
    """
    defines the PLAPT model
    """
    def __init__(self, prot_hidden, lig_hidden):
        super().__init__()
        self.protein_layer = Linear(in_features=1024, out_features=prot_hidden)
        self.ligand_layer = Linear(in_features=768, out_features=lig_hidden)
        self.final_layer = Linear(in_features=prot_hidden+lig_hidden, out_features=1)

    def forward(self, prot, lig):
        embedded_prot = self.protein_layer(prot)
        embedded_lig = self.ligand_layer(lig)
        return self.final_layer(torch.concat(tensors=[embedded_prot, embedded_lig], dim=1))


def validate(val_dl: Type[DataLoader], model: Type[PLAPT], loss_fn: Type[torch.nn.MSELoss], device: Type[torch.device]):
    """
    evaluates model performance on the validation dataset
    """
    with torch.no_grad():
        model.eval()
        running_loss = 0
        for prot, lig, aff in tqdm(val_dl):
            prot = prot.to(device)
            lig = lig.to(device)
            aff = aff.to(device, dtype=torch.float32)

            score = model(prot, lig)
            loss_val = loss_fn(input=score.squeeze(1), target=aff)
            running_loss += loss_val.item()

        running_loss /= len(val_dl)
    return running_loss

            
def train(
        model: Type[PLAPT], 
        train_dl: Type[DataLoader], 
        val_dl: Type[DataLoader], 
        device: Type[torch.device], 
        loss_fn: Type[torch.nn.MSELoss], 
        optim: Type[torch.optim.Adam], 
        num_epochs: int, 
        result_folder: Path, 
        model_save_name: str,
        patience: int,
        ) -> None:

    patience_counter = 0
    best_val_loss = float("inf")
    results = {
        "train_losses": [],
        "val_losses": []
    }

    for epoch_idx in range(num_epochs):
        if patience_counter < patience:
            model.train()
            running_loss = 0
            for prot, lig, aff in tqdm(train_dl):
                prot = prot.to(device)
                lig = lig.to(device)
                aff = aff.to(device, dtype=torch.float32)
                score = model(prot, lig)
                loss_val = loss_fn(input=score.squeeze(1), target=aff)

                optim.zero_grad()
                loss_val.backward()
                optim.step()

                running_loss += loss_val.item()
                print(running_loss)
            running_loss /= len(train_dl)
            results["train_losses"].append(running_loss)
            print(f"epoch: {epoch_idx}, loss: {running_loss}")

            val_loss = validate(val_dl=val_dl, model=model, loss_fn=loss_fn, device=device)
            results["val_losses"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
            else:
                patience_counter += 1

            


    with open(result_folder.joinpath(model_save_name+"_results.json")) as opened_json:
        json.dump(results, opened_json)

    



def main():
    args = parse_cla()
    device = torch.device('cuda')
    train_dl, val_dl, test_dl = create_dataloaders(
        csv_folder=args.csv_folder, 
        data_folder=args.data_folder, 
        batch_size=args.batch_size
        )
    model = PLAPT(prot_hidden=512, lig_hidden=512).to(device)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    train(
        model=model, 
        train_dl=train_dl, 
        val_dl=val_dl,
        val_dl=val_dl, 
        device=device, 
        loss_fn=loss_fn, 
        optim=optim, 
        num_epochs=args.num_epochs
        )



if __name__ == "__main__":
    main()
