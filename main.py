"""
Main script that trains, validates, and evaluates
various models including AASIST and the Hybrid AASIST_ARNet.

MODIFIED: Supports HybridFusionModel with dual inputs (raw_audio, handcrafted_features).
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train, # Expects raw audio + MFCC
                        Dataset_ASVspoof2019_devNeval, genSpoof_list) 
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """
    Make PyTorch DataLoaders for train / developement / evaluation, 
    modified to support dual input (raw audio + MFCCs).
    """
    # Define paths
    track = config["track"]
    trn_database_path = database_path + "/wav/" + track + "/ASVspoof2019_" + track + "_train/"
    dev_database_path = config["dev_database_path"] + "/wav/" + track + "/ASVspoof2019_" + track + "_dev/"
    eval_database_path = config["eval_database_path"] + "/wav/" + track + "/ASVspoof2019_" + track + "_eval/"
    trn_list_path = database_path + "/ASVspoof2019_" + track + "_cm_protocols/ASVspoof2019." + track + ".cm.train.trn.txt"
    dev_trial_path = config["dev_database_path"] + "/ASVspoof2019_" + track + "_cm_protocols/ASVspoof2019." + track + ".cm.dev.trl.txt"
    eval_trial_path = config["eval_database_path"] + "/ASVspoof2019_" + track + "_cm_protocols/" + config["protocol"]
    
    # --- Training Loader ---
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
    print("no. training files:", len(file_train))

    # Dataset_ASVspoof2019_train returns (raw_audio, mfccs, label)
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # --- Development Loader ---
    _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
    print("no. validation files:", len(file_dev))

    # Dataset_ASVspoof2019_devNeval returns (raw_audio, mfccs, utt_id)
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev, 
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # --- Evaluation Loader ---
    file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, 
                                            base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """
    Perform evaluation and save the score to a file, modified for dual input.
    """
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    
    # Assuming dev/eval data loader returns (raw_audio, mfccs, utt_id)
    for batch_x_raw, batch_x_mfcc, utt_id in data_loader:
        batch_x_raw = batch_x_raw.to(device)
        batch_x_mfcc = batch_x_mfcc.to(device)
        with torch.no_grad():
            # Pass both inputs to the hybrid model
            _, batch_out = model(batch_x_raw, batch_x_mfcc) 
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model: nn.Module,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """
    Train the model for one epoch, modified for dual input.
    """
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # train data loader returns (raw_audio, mfccs, label)
    for batch_x_raw, batch_x_mfcc, batch_y in trn_loader:
        batch_size = batch_x_raw.size(0)
        num_total += batch_size
        ii += 1
        
        batch_x_raw = batch_x_raw.to(device)
        batch_x_mfcc = batch_x_mfcc.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Pass BOTH inputs to the hybrid model
        _, batch_out = model(batch_x_raw, batch_x_mfcc, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


def validate(
    dev_loader: DataLoader,
    model: nn.Module,
    device: torch.device) -> Tuple[float, float]:
    """
    Validate the model.
    """
    running_loss = 0
    num_total = 0.0
    
    model.eval()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    with torch.no_grad():
        # dev data loader returns (raw_audio, mfccs, label)
        for batch_x_raw, batch_x_mfcc, batch_y in dev_loader:
            batch_size = batch_x_raw.size(0)
            num_total += batch_size
            
            batch_x_raw = batch_x_raw.to(device)
            batch_x_mfcc = batch_x_mfcc.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # Pass BOTH inputs to the hybrid model
            _, batch_out = model(batch_x_raw, batch_x_mfcc)
            batch_loss = criterion(batch_out, batch_y)
            running_loss += batch_loss.item() * batch_size

    running_loss /= num_total
    return running_loss, 0 # Return (loss, EER)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    comment = args.comment
    
    # set seed
    set_seed(args.seed, config)

    # define model name
    model_tag = "AASIST_ARNet_Hybrid"
    if comment:
        model_tag = model_tag + "-{}".format(comment)
    
    # set model path
    model_save_path = Path(config["model_path"]) / model_tag
    os.makedirs(model_save_path, exist_ok=True)
    
    # copy config file
    copy(args.config, model_save_path / "config.json")
    
    # make output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    
    # get data loader
    trn_loader, dev_loader, eval_loader = get_loader(
        config["database_path"], args.seed, config)

    optim_config["steps_per_epoch"] = len(trn_loader)
    
    # load model
    model_module = import_module(model_config["architecture"]) # Imports AASIST_ARNet_Hybrid.py
    model = model_module.Model(model_config).to(device)

    # Define optimizer, scheduler, and SWA
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    
    if "swa_start" in optim_config and optim_config["swa_start"] is not None:
        swa_scheduler = SWA(optimizer, swa_start=optim_config["swa_start"], swa_freq=optim_config["swa_freq"], swa_lr=optim_config["swa_lr"])
    else:
        swa_scheduler = None


    num_epochs = config["num_epochs"]
    best_dev_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        running_loss = train_epoch(trn_loader, model, optimizer, device, scheduler, config)
        # Validation
        dev_loss, _ = validate(dev_loader, model, device)
        
        # Save the best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), model_save_path / "best_model.pth")
            print(f"Epoch {epoch+1}: Model saved to {model_save_path}")

        print(f"Epoch {epoch+1}: Train Loss: {running_loss:.4f}, Dev Loss: {dev_loss:.4f}")


    # Evaluation
    if args.eval or args.eval_model_weights:
        # Load best model weights
        model_weights = args.eval_model_weights or (model_save_path / "best_model.pth").as_posix()
        if os.path.exists(model_weights):
            model.load_state_dict(torch.load(model_weights, map_location=device))
            print(f"Loaded model weights from {model_weights}")
        else:
            print(f"Error: Model weights not found at {model_weights}. Cannot run evaluation.")
            return

        # Produce scores
        save_path = Path(output_dir) / f"{model_tag}_eval_scores.txt"
        trial_path = config["eval_database_path"] + "/ASVspoof2019_" + track + "_cm_protocols/" + config["protocol"]
        produce_evaluation_file(eval_loader, model, device, save_path.as_posix(), trial_path)
        
        # Calculate EER and tDCF
        EER, tDCF = calculate_tDCF_EER(save_path, trial_path)
        print(f"Evaluation Results: EER={EER:.4f}, tDCF={tDCF:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())