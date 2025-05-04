import json
from tqdm import tqdm
from torch.nn import Linear
from torch.utils.data import DataLoader, Dataset
import csv
import argparse
import torch
from pathlib import Path
from typing import List, Tuple, Type, Dict
from sklearn.metrics import r2_score
import random
import numpy as np


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    # folder with the processed dataset files
    parser.add_argument("-data_folder", type=Path)
    # folder with dataset csv files
    parser.add_argument("-csv_folder", type=Path)
    # number of iterations for the model to process before the gradient is measured
    parser.add_argument("-batch_size", type=int, default=64)
    # learning rate: the proportion of the gradient that is used for parameter updates
    parser.add_argument("-lr", type=float, default=1e-3)
    # total number, including epochs that have alrady been trained
    parser.add_argument("-num_epochs", type=int, default=8)
    # folder to save model and performance to
    parser.add_argument("-result_folder", type=Path)
    # name of the model file
    parser.add_argument("-model_save_name", type=str, default="model_1.pth.tar")
    # amount of epochs to train past the model not improving on evaluation loss
    parser.add_argument("-patience", type=int, default=5)
    # if true, training will resume from latest epoch saved, otherwise 
    parser.add_argument("-resume", action="store_true", default=False)
    # size of the hidden state of the protein embedding layer
    parser.add_argument("-prot_hidden", type=int, default=512)
    # size of the hidden state of the ligand embedding layer
    parser.add_argument("-lig_hidden", type=int, default=512)
    return parser.parse_args()


def read_ds_csvs(csv_folder: Path) -> Tuple[List]:
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
    def __init__(self, prot_hidden: int, lig_hidden: int) -> None:
        super().__init__()
        self.protein_layer = Linear(in_features=1024, out_features=prot_hidden)
        self.ligand_layer = Linear(in_features=768, out_features=lig_hidden)
        self.final_layer = Linear(in_features=prot_hidden+lig_hidden, out_features=1)

    def forward(self, prot: torch.tensor, lig: torch.tensor) -> torch.tensor:
        # embed protein
        embedded_prot = self.protein_layer(prot)
        # embed ligand
        embedded_lig = self.ligand_layer(lig)
        # predict binding affinity from concatenated protein and ligand vectors
        return self.final_layer(torch.concat(tensors=[embedded_prot, embedded_lig], dim=1))


def eval(
        data_loader: Type[DataLoader], 
        model: Type[PLAPT], 
        loss_fn: Type[torch.nn.MSELoss], 
        device: Type[torch.device]
        ) -> Tuple[float]:
    """
    evaluates model performance
    """
    with torch.no_grad():
        # lists to append results to in order for r2 calculation
        pred = []
        ground_truth = []

        # turn off dropout, batch norm, etc
        model.eval()
        eval_loss = 0

        for prot, lig, aff in tqdm(data_loader):
            # put tensors on device
            prot = prot.to(device)
            lig = lig.to(device)
            aff = aff.to(device, dtype=torch.float32)

            # predict binding affinity
            score = model(prot, lig)
            # calculate loss
            loss_val = loss_fn(input=score.squeeze(1), target=aff)
            # add loss form batch
            eval_loss += loss_val.item()

            pred += [x.item() for x in aff]
            ground_truth += [x.item() for x in score]

        # average loss by the number of batches
        eval_loss /= len(data_loader)
        # calculate r2
        r2 = r2_score(y_true=ground_truth, y_pred=pred)

    return eval_loss, r2

            
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
    results: Dict = None,
    resume_epoch: int = None
    ) -> None:

    # counter to keep track of patience
    patience_counter = 0
    best_val_loss = float("inf")

    # if no results were loaded, a new dictionary is created
    if results == None:
        results = {
            "train_losses": [],
            "val_losses": [],
            "val_r2s": [],
        }
    # reset start epoch if resuming
    if resume_epoch != None:
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 0
    
    if resume_epoch != None:
        print(f"resuming training from epoch: {start_epoch}")

    for epoch_idx in range(start_epoch, num_epochs):
        if patience_counter == patience:
            # early stopping
            break

        model.train()
        epoch_loss = 0

        for prot, lig, aff in tqdm(train_dl):
            # put tensors on device
            prot = prot.to(device)
            lig = lig.to(device)
            aff = aff.to(device, dtype=torch.float32)

            # predict binding affinity
            score = model(prot, lig)
            # score prediction
            loss_val = loss_fn(input=score.squeeze(1), target=aff)

            # reset gradient values if there are any
            optim.zero_grad()
            # calculate gradient
            loss_val.backward()
            # update parameters
            optim.step()

            # add loss for batch
            epoch_loss += loss_val.item()

        # average loss by the number of batches
        epoch_loss /= len(train_dl)
        results["train_losses"].append(epoch_loss)

        # evaluate model on validation set
        val_loss, val_r2 = eval(data_loader=val_dl, model=model, loss_fn=loss_fn, device=device)
        results["val_losses"].append(val_loss)
        results["val_r2s"].append(val_r2)
        print(f"epoch: {epoch_idx}, train loss: {epoch_loss}, val loss: {val_loss}, val r2: {val_r2}")


        if val_loss < best_val_loss:
            # if best performing model, save checkpoint
            # with 'best' in the name
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optim,
                epoch=epoch_idx,
                metrics_history=results,
                checkpoint_dir=result_folder,
                best=True
            )
        else:
            # otherwise just save checkpoint
            patience_counter += 1
            save_checkpoint(
                model=model,
                optimizer=optim,
                epoch=epoch_idx,
                metrics_history=results,
                checkpoint_dir=result_folder,
                best=False
            )

    # save training results
    with open(result_folder.joinpath(model_save_name+"_results.json"), mode="w") as opened_json:
        json.dump(results, opened_json)


def set_seed(seed: int) -> None:
    """
    sets seed for torch, numpy and random
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(
        model: Type[PLAPT], 
        optimizer: Type[torch.optim.Adam], 
        epoch: int, 
        metrics_history: Dict, 
        checkpoint_dir: Path, 
        best: bool
        ):
    """
    save all necessary training state for resuming later
    """
    print('saving...')
    json_checkpoint = {
        'metrics_history': metrics_history,         
        'epoch': epoch,
    }
    tensor_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if best:
        torch.save(tensor_checkpoint, checkpoint_dir.joinpath(f'best_checkpoint_tensor_epoch_{epoch}.pt'))
        with open(checkpoint_dir.joinpath(f'best_checkpoint_json_epoch_{epoch}.pt'), mode="w") as opened_json:
            json.dump(json_checkpoint, opened_json)
    else:
        torch.save(tensor_checkpoint, checkpoint_dir.joinpath(f'checkpoint_tensor_epoch_{epoch}.pt'))
        with open(checkpoint_dir.joinpath(f'checkpoint_json_epoch_{epoch}.pt'), mode="w") as opened_json:
            json.dump(json_checkpoint, opened_json)


def load_checkpoint(
        checkpoint_dir: Path, 
        model: Type[PLAPT], 
        optimizer: Type[torch.optim.Adam], 
        device: Type[torch.device], 
        best_only: bool
        ):
    """
    loads the latest epoch inside checkpoint_dir folder
    """
    if best_only:
        tensor_list = [x for x in checkpoint_dir.glob("best_checkpoint_tensor_epoch_*.pt")]
        tensor_list.sort(reverse=True)

        json_list = [x for x in checkpoint_dir.glob("best_checkpoint_json_epoch_*.pt")]
        json_list.sort(reverse=True)
    else:
        tensor_list = [x for x in checkpoint_dir.glob("*checkpoint_tensor_epoch_*.pt")]
        tensor_list.sort(reverse=True)

        json_list = [x for x in checkpoint_dir.glob("*checkpoint_json_epoch_*.pt")]
        json_list.sort(reverse=True)

    tensor_checkpoint = torch.load(tensor_list[0], map_location=device)
    
    with open(json_list[0]) as opened_json:
        json_checkpoint = json.load(opened_json)

    # restore model parameters
    model.load_state_dict(tensor_checkpoint['model_state_dict'])
    
    # restore optimizer state
    optimizer.load_state_dict(tensor_checkpoint['optimizer_state_dict'])
    
    # return the epoch to resume from and metrics history
    return json_checkpoint['epoch'], json_checkpoint['metrics_history']


def save_test_results(
        save_folder: Path, 
        model_save_name: str, 
        test_r2: float, 
        test_loss: float
        ) -> None:
    """
    saves test performance
    """
    with open(save_folder.joinpath(f"{model_save_name}_results.json"), mode="w") as opened_json:
        save_obj = {"test_r2": test_r2, "test_loss": test_loss}
        json.dump(save_obj, opened_json)


def main():
    set_seed(42)
    args = parse_cla()
    device = torch.device('cuda')
    train_dl, val_dl, test_dl = create_dataloaders(
        csv_folder=args.csv_folder, 
        data_folder=args.data_folder, 
        batch_size=args.batch_size
        )
    model = PLAPT(prot_hidden=args.prot_hidden, lig_hidden=args.lig_hidden).to(device)
    
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    
    if args.resume:
        epoch_idx, results = load_checkpoint(
            checkpoint_dir=args.result_folder,
            model=model,
            optimizer=optim,
            device=device,
            best_only=False
        )
        train(
            model=model, 
            train_dl=train_dl, 
            val_dl=val_dl,
            device=device, 
            loss_fn=loss_fn, 
            optim=optim, 
            num_epochs=args.num_epochs,
            result_folder=args.result_folder,
            model_save_name=args.model_save_name,
            patience=args.patience,
            results=results,
            resume_epoch=epoch_idx
            )
        _ = load_checkpoint(
            checkpoint_dir=args.result_folder,
            model=model,
            optimizer=optim,
            device=device,
            best_only=True
            )
        test_loss, test_r2 = eval(
            data_loader=test_dl,
            model=model,
            loss_fn=loss_fn,
            device=device
        )
        save_test_results(
            args.result_folder, 
            model_save_name=args.model_save_name, 
            test_r2=test_r2, 
            test_loss=test_loss
            )
        print(f"Test loss: {test_loss}, test r2: {test_r2}")

    else:
        train(
            model=model, 
            train_dl=train_dl, 
            val_dl=val_dl,
            device=device, 
            loss_fn=loss_fn, 
            optim=optim, 
            num_epochs=args.num_epochs,
            result_folder=args.result_folder,
            model_save_name=args.model_save_name,
            patience=args.patience,
            )
        _ = load_checkpoint(
            checkpoint_dir=args.result_folder,
            model=model,
            optimizer=optim,
            device=device,
            best_only=True
            )
        test_loss, test_r2 = eval(
            data_loader=test_dl,
            model=model,
            loss_fn=loss_fn,
            device=device
        )
        save_test_results(
            save_folder=args.result_folder, 
            model_save_name=args.model_save_name, 
            test_r2=test_r2, 
            test_loss=test_loss
            )
        print(f"Test loss: {test_loss}, test r2: {test_r2}")


if __name__ == "__main__":
    main()
