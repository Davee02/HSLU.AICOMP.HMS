import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if bool(os.environ.get("KAGGLE_URL_BASE", "")):
    # running on Kaggle
    sys.path.insert(0, "/kaggle/input/hsm-source-files")
else:
    # running locally
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.eeg_processor import EEGDataProcessor
from src.datasets.multi_spectrogram import MultiSpectrogramDataset
from src.models.base_cnn import BaseCNN
from src.utils.constants import Constants
from src.utils.k_folds_creator import KFoldCreator
from src.utils.utils import (
    get_models_save_path,
    get_processed_data_dir,
    get_raw_data_dir,
    set_seeds,
)


class CFG:
    def __init__(self, args):
        self.seed = args.seed
        self.n_splits = 5
        self.data_path = get_raw_data_dir()
        self.train_eeg_spec_path = get_processed_data_dir() / "eeg_spectrograms" / "train" / "cwt"

        self.model_name = args.model_name
        self.target_size = 6

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr

        # Base size of each channel
        self.img_size = (128, 256)


def get_dataloaders(df, fold_id, cfg, targets):
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    valid_df = df[df["fold"] == fold_id].reset_index(drop=True)

    train_dataset = MultiSpectrogramDataset(
        train_df, targets, cfg.data_path, cfg.img_size, cfg.train_eeg_spec_path, mode="train"
    )
    valid_dataset = MultiSpectrogramDataset(
        valid_df, targets, cfg.data_path, cfg.img_size, cfg.train_eeg_spec_path, mode="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def run_training(df, data_preparation_vote_method, cfg, targets, use_wandb=True, wandb_project=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_oof_preds = []
    all_oof_labels = []

    for fold in range(cfg.n_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}")
        print(f"{'='*50}")

        # W&B configuration
        if use_wandb:
            config = {
                # Model
                "architecture": cfg.model_name,
                "pretrained": True,
                # Data
                "fold": fold,
                "features": "multi_spectrograms",
                "window_selection": data_preparation_vote_method,
                "img_size": cfg.img_size,
                # Training
                "optimizer": "AdamW",
                "learning_rate": cfg.lr,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "seed": cfg.seed,
                "scheduler": "CosineAnnealingLR",
            }

            wandb.init(
                project=wandb_project or "hms-aicomp-cnn-multispec",
                name=f"{cfg.model_name}-multispec-fold{fold}",
                tags=[f"fold{fold}", cfg.model_name],
                config=config,
            )

        # init model
        model = BaseCNN(cfg.model_name, pretrained=True, num_classes=cfg.target_size)
        model.to(device)

        # setup optimizer, scheduler and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        loss_fn = nn.KLDivLoss(reduction="batchmean")

        train_loader, valid_loader = get_dataloaders(df, fold, cfg, targets)

        best_val_loss = float("inf")
        best_model_path = None

        # training loop
        for epoch in range(cfg.epochs):
            print(f"\n--- Epoch {epoch+1}/{cfg.epochs} ---")

            model.train()
            train_loss = 0
            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = loss_fn(log_probs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

                if use_wandb:
                    wandb.log({"train/loss": loss.item()})

            train_loss /= len(train_loader.dataset)

            # validation phase
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for images, labels in tqdm(valid_loader, desc="Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    log_probs = F.log_softmax(outputs, dim=1)
                    loss = loss_fn(log_probs, labels)
                    valid_loss += loss.item() * images.size(0)

            valid_loss /= len(valid_loader.dataset)

            epoch_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, LR = {epoch_lr:.6f}")

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/epoch_loss": train_loss,
                        "val/loss": valid_loss,
                        "val/kl_div": valid_loss,
                        "train/epoch_lr": epoch_lr,
                    }
                )

            # save best model
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_model_path = (
                    get_models_save_path()
                    / "multi_spec_cnn"
                    / cfg.model_name
                    / data_preparation_vote_method
                    / f"best_model_fold{fold}.pth"
                )
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

            scheduler.step()

        # generate OOF predictions
        print(f"\n--- Generating OOF predictions for fold {fold} ---")
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            fold_oof_preds = []
            fold_oof_labels = []

            with torch.no_grad():
                for images, labels in tqdm(valid_loader, desc=f"OOF Prediction Fold {fold}"):
                    images = images.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1).cpu()

                    fold_oof_preds.append(probs)
                    fold_oof_labels.append(labels.cpu())

            all_oof_preds.append(torch.cat(fold_oof_preds).numpy())
            all_oof_labels.append(torch.cat(fold_oof_labels).numpy())
            print(f"Finished OOF predictions for fold {fold}")
        else:
            raise ValueError("Best model path is None, cannot generate OOF predictions.")

        # log to W&B
        if use_wandb:
            wandb.summary["best_val_kl_div"] = best_val_loss

            if best_model_path:
                artifact = wandb.Artifact(f"{cfg.model_name}-fold{fold}", type="model")
                artifact.add_file(best_model_path)
                wandb.log_artifact(artifact)
                print(f"\nLogged artifact for fold {fold} with best validation loss: {best_val_loss:.4f}")
            else:
                print("\nNo best model was saved during training for this fold.")

            wandb.finish()

    # calculate overall OOF score
    if all_oof_preds and all_oof_labels:
        print(f"\n{'='*50}")
        print("Calculating final OOF score...")
        print(f"{'='*50}")

        final_oof_preds = np.concatenate(all_oof_preds)
        final_oof_labels = np.concatenate(all_oof_labels)

        oof_preds_tensor = torch.tensor(final_oof_preds, dtype=torch.float32)
        oof_labels_tensor = torch.tensor(final_oof_labels, dtype=torch.float32)

        log_oof_preds_tensor = torch.log(oof_preds_tensor)

        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        overall_oof_score = kl_loss_fn(log_oof_preds_tensor, oof_labels_tensor).item()

        print(f"\nOverall OOF KL Score: {overall_oof_score:.4f}")
    else:
        print("\nCould not calculate OOF score because no predictions were generated.")
        overall_oof_score = None

    return overall_oof_score


def main():
    parser = argparse.ArgumentParser(description="Train CNN on Kaggle + EEG spectrograms")

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pretrained model name (e.g., tf_efficientnet_b0_ns, resnet50, efficientnet_b1)",
    )

    # Data parameters
    parser.add_argument(
        "--vote_method",
        type=str,
        default="max_vote_window",
        choices=["max_vote_window", "sum_and_normalize"],
        help="Method for aggregating predictions from overlapping windows",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")

    # Cross-validation parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Logging parameters
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="hms-aicomp", help="W&B project name")

    args = parser.parse_args()

    cfg = CFG(args)

    set_seeds(cfg.seed)

    print(f"\n{'='*50}")
    print("Training Configuration")
    print(f"{'='*50}")
    print(f"Model: {cfg.model_name}")
    print(f"Data path: {cfg.data_path}")
    print(f"Vote method: {args.vote_method}")
    print(f"Image size: {cfg.img_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Number of folds: {cfg.n_splits}")
    print(f"Random seed: {cfg.seed}")
    print(f"W&B logging: {args.wandb}")
    print(f"{'='*50}\n")

    # W&B login
    if args.wandb:
        wandb.login(verify=True)

    # prepare data
    processor = EEGDataProcessor(raw_data_path=cfg.data_path, processed_data_path=get_processed_data_dir())

    print("Preparing data and creating folds...")
    train_df = processor.process_data(vote_method=args.vote_method, skip_parquet=True)
    print(f"Train shape: {train_df.shape}")
    print(f"Targets: {list(Constants.TARGETS)}")

    fold_creator = KFoldCreator(n_splits=cfg.n_splits, seed=cfg.seed)
    train_df = fold_creator.create_folds(train_df, stratify_col="expert_consensus", group_col="patient_id")

    # run training
    overall_oof_score = run_training(
        df=train_df,
        data_preparation_vote_method=args.vote_method,
        cfg=cfg,
        targets=Constants.TARGETS,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # print final results
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    if overall_oof_score is not None:
        print(f"Final Overall OOF KL Score: {overall_oof_score:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
