#!/usr/bin/env python
import os
import argparse
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
from model import UNetWithOutcome4
from skimage.metrics import (
    structural_similarity as ssim_metric,
    peak_signal_noise_ratio as psnr_metric,
)
from PIL import Image


class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if tensor.shape[0] == len(self.mean):
            return transforms.functional.normalize(tensor, self.mean, self.std)
        elif tensor.shape[0] == 1:
            return transforms.functional.normalize(
                tensor, [self.mean[0]], [self.std[0]]
            )
        else:
            raise ValueError(
                f"Expected tensor with 1 or 3 channels, got {tensor.shape[0]} channels."
            )


class ScarDiceLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, pred_mask, true_mask):
        true_mask = true_mask.squeeze(1)  # (B, H, W)
        pred_mask = pred_mask.squeeze(1)  # (B, H, W)
        pred_flat = pred_mask.view(pred_mask.size(0), -1)
        true_flat = true_mask.view(true_mask.size(0), -1)
        intersection = (pred_flat * true_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + true_flat.sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return (1 - dice).mean()


class Phase1Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        patient_id = row["ID"]
        pre_img = Image.open(row["Pre"]).convert("RGB")
        duration_img = Image.open(row["Pre_DurationTime"]).convert("L")
        avgforce_img = Image.open(row["Pre_AverageForce"]).convert("L")
        maxtemp_img = Image.open(row["Pre_MaxTemperature"]).convert("L")
        maxpower_img = Image.open(row["Pre_MaxPower"]).convert("L")
        post_img = Image.open(row["Post"]).convert("RGB")
        mask_img = Image.open(row["Mask"]).convert("L")

        if self.transform:
            pre_img = self.transform(pre_img)
            duration_img = self.transform(duration_img)
            avgforce_img = self.transform(avgforce_img)
            maxtemp_img = self.transform(maxtemp_img)
            maxpower_img = self.transform(maxpower_img)
            post_img = self.transform(post_img)
            mask_img = self.transform(mask_img)
        input_tensor = torch.cat(
            [pre_img, duration_img, avgforce_img, maxtemp_img, maxpower_img], dim=0
        )

        outcome = float(row["PrimaryAAOutcome"])
        outcome = torch.tensor(outcome, dtype=torch.float)
        return input_tensor, post_img, mask_img, patient_id, outcome


class Phase2Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.transform = transform
        self.patient_groups = dataframe.groupby("ID")
        self.patient_ids = list(self.patient_groups.groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = self.patient_groups.get_group(patient_id).sort_values(by="View")

        inputs, targets, masks = [], [], []
        for _, row in data.iterrows():
            pre_img = Image.open(row["Pre"]).convert("RGB")
            duration_img = Image.open(row["Pre_DurationTime"]).convert("L")
            avgforce_img = Image.open(row["Pre_AverageForce"]).convert("L")
            maxtemp_img = Image.open(row["Pre_MaxTemperature"]).convert("L")
            maxpower_img = Image.open(row["Pre_MaxPower"]).convert("L")
            post_img = Image.open(row["Post"]).convert("RGB")
            mask_img = Image.open(row["Mask"]).convert("L")

            if self.transform:
                pre_img = self.transform(pre_img)
                duration_img = self.transform(duration_img)
                avgforce_img = self.transform(avgforce_img)
                maxtemp_img = self.transform(maxtemp_img)
                maxpower_img = self.transform(maxpower_img)
                post_img = self.transform(post_img)
                mask_img = self.transform(mask_img)

            input_tensor = torch.cat(
                [pre_img, duration_img, avgforce_img, maxtemp_img, maxpower_img], dim=0
            )
            inputs.append(input_tensor)
            targets.append(post_img)
            masks.append(mask_img)
        inputs = torch.stack(inputs)  # (num_views, 7, H, W)
        targets = torch.stack(targets)  # (num_views, 3, H, W)
        masks = torch.stack(masks)  # (num_views, 1, H, W)
        outcome = float(data["PrimaryAAOutcome"].iloc[0])
        outcome = torch.tensor(outcome, dtype=torch.float)
        if outcome.isnan():
            print(f"NaN detected in outcome for patient {patient_id}")
        return inputs, targets, masks, patient_id, outcome


def compute_metrics(outputs, targets):
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    mse_val = ((outputs_np - targets_np) ** 2).mean()
    psnr_vals = []
    ssim_vals = []
    for i in range(outputs_np.shape[0]):
        out_img = np.clip(outputs_np[i].transpose(1, 2, 0), 0, 1)
        target_img = np.clip(targets_np[i].transpose(1, 2, 0), 0, 1)
        psnr_vals.append(psnr_metric(target_img, out_img, data_range=1.0))
        ssim_vals.append(
            ssim_metric(
                target_img, out_img, win_size=7, channel_axis=-1, data_range=1.0
            )
        )
    return mse_val, np.mean(psnr_vals), np.mean(ssim_vals)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UNet for simulation and outcome prediction with 5-fold CV"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/dataset_filtered.csv",
        help="Path to CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--phase2_lr", type=float, default=1e-5, help="Phase 2 LR")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l1",
        help="Loss function for simulation (l1 or mse)",
    )
    parser.add_argument(
        "--num_views", type=int, default=6, help="Number of views per patient"
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2], help="Training phase: 1 or 2"
    )
    return parser.parse_args()


def train(args, device):
    df = pd.read_csv(args.csv_path)
    patient_ids = df["ID"].unique()
    print(f"Total patients: {len(patient_ids)}")

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dataset = Phase1Dataset if args.phase == 1 else Phase2Dataset

    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"\nFold {fold} starting...")
        # if fold != 0:
        #     continue
        if os.path.exists(
            os.path.join(
                args.out,
                f"validation_log_phase{args.phase}_fold{fold}_{args.phase2_lr}.csv",
            )
        ):
            print(f"Fold {fold} already trained, skipping...")
            continue
        train_ids = patient_ids[train_idx]
        val_ids = patient_ids[val_idx]
        train_df = df[df["ID"].isin(train_ids)]
        val_df = df[df["ID"].isin(val_ids)]
        print(f"Fold {fold} | Train: {len(train_ids)} | Validation: {len(val_ids)}")

        # Datasets and loaders
        train_dataset = dataset(train_df, transform=transform)
        val_dataset = dataset(val_df, transform=transform)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Model setup
        if args.phase == 1:
            model = UNetWithOutcome4(
                out_channels=3, num_views=1, freeze_encoder=False
            )
        else:
            model = UNetWithOutcome4(
                out_channels=3,
                num_views=args.num_views,
                freeze_encoder=True,
            )
        model = model.to(device)

        # Load Phase 1 checkpoint for Phase 2
        if args.phase == 2:
            checkpoint_path = os.path.join(args.out, f"best_phase1_fold{fold}.pth")
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=device)
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("classifier_heads")
                }
                model.load_state_dict(state_dict, strict=False)
                print(
                    f"Loaded Phase 1 checkpoint from {checkpoint_path} for fold {fold}"
                )
                for i in range(model.num_views):
                    init.uniform_(model.classifier_heads[i][2].weight)
                    init.zeros_(model.classifier_heads[i][2].bias)
                    init.xavier_uniform_(model.classifier_heads[i][3].weight)
                    if model.classifier_heads[i][3].bias is not None:
                        model.classifier_heads[i][3].bias.data.zero_()
            else:
                print(
                    f"Phase 1 checkpoint for fold {fold} not found. Using random init."
                )

        # Loss functions
        sim_criterion = nn.MSELoss() if args.loss.lower() == "mse" else nn.L1Loss()
        dice_loss_fn = ScarDiceLoss(device)
        cls_criterion = nn.BCEWithLogitsLoss()

        if args.phase == 2:
            args.lr = args.phase2_lr  # Lower learning rate for Phase 2
            args.epochs = 50
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Logging setup
        if args.phase == 1:
            log_csv = os.path.join(
                args.out, f"validation_log_phase{args.phase}_fold{fold}.csv"
            )
        else:
            log_csv = os.path.join(
                args.out,
                f"validation_log_phase{args.phase}_fold{fold}_{args.phase2_lr}.csv",
            )
        if args.phase == 1:
            log_fields = [
                "epoch",
                "TrainLoss",
                "TrainMSE",
                "TrainDice",
                "ValLoss",
                "MSE",
                "PSNR",
                "SSIM",
                "Dice",
                "SSIM+Dice",
            ]
        else:
            log_fields = ["epoch", "ValLoss", "Accuracy", "AUC"]
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()

        best_metric = None
        for epoch in range(args.epochs):
            model.train()
            running_loss = running_dice = running_mse = 0.0
            for sample in train_loader:
                if args.phase == 1:
                    inputs, targets, masks, pid, outcome = sample
                else:
                    inputs, targets, masks, pid, outcome = sample
                inputs = inputs.to(device)
                targets = targets.to(device)
                outcome = outcome.to(device).unsqueeze(1)
                masks = masks.to(device)
                optimizer.zero_grad()
                sim_outputs, outcome_pred, mask_outs = model(inputs)
                if args.phase == 1:
                    loss_pixel = sim_criterion(sim_outputs, targets)
                    loss_dice = dice_loss_fn(mask_outs, masks)
                    loss = loss_pixel + 0.1 * loss_dice
                    running_dice += loss_dice.item() * inputs.size(0)
                    running_mse += loss_pixel.item() * inputs.size(0)
                else:
                    if (
                        torch.isnan(outcome_pred).any()
                        or torch.isnan(outcome_pred).any()
                    ):
                        print("NaN detected in outcome_pred, skipping batch")
                        continue
                    loss = cls_criterion(outcome_pred, outcome)
                    if torch.isnan(loss):
                        print("NaN detected in loss, skipping batch")
                        continue
                loss.backward()
                if args.phase == 2:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            if args.phase == 1:
                epoch_dice = running_dice / len(train_loader.dataset)
                epoch_mse = running_mse / len(train_loader.dataset)
                print(
                    f"Fold {fold} Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f} | MSE: {epoch_mse:.4f} | Dice: {epoch_dice:.4f}"
                )
            else:
                print(
                    f"Fold {fold} Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f}"
                )

            # Validation
            model.eval()
            if args.phase == 1:
                total_loss = total_mse = total_psnr = total_ssim = total_dice = 0.0
                num_batches = 0
                with torch.no_grad():
                    for inputs, targets, masks, pid, outcome in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        masks = masks.to(device)
                        sim_outputs, _, mask_outs = model(inputs)
                        loss = sim_criterion(sim_outputs, targets)
                        total_loss += loss.item() * inputs.size(0)
                        mse_val, psnr_val, ssim_val = compute_metrics(
                            sim_outputs, targets
                        )
                        total_mse += mse_val
                        total_psnr += psnr_val
                        total_ssim += ssim_val
                        dice_score = 1 - dice_loss_fn(mask_outs, masks).item()
                        total_dice += dice_score
                        num_batches += 1
                avg_loss = total_loss / len(val_loader.dataset)
                avg_mse = total_mse / num_batches
                avg_psnr = total_psnr / num_batches
                avg_ssim = total_ssim / num_batches
                avg_dice = total_dice / num_batches
                combined_metric = avg_ssim + avg_dice
                print(
                    f"Fold {fold} Val Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | Dice: {avg_dice:.4f} | SSIM+Dice: {combined_metric:.4f}"
                )
                log_dict = {
                    "epoch": epoch + 1,
                    "TrainLoss": epoch_loss,
                    "TrainMSE": epoch_mse,
                    "TrainDice": epoch_dice,
                    "ValLoss": avg_loss,
                    "MSE": avg_mse,
                    "PSNR": avg_psnr,
                    "SSIM": avg_ssim,
                    "Dice": avg_dice,
                    "SSIM+Dice": combined_metric,
                }
                metric = combined_metric
            else:
                total_loss = total_correct = total_samples = 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for inputs, targets, masks, pid, outcome in val_loader:

                        inputs = inputs.to(device)
                        outcome = outcome.to(device).unsqueeze(1)
                        _, outcome_pred, _ = model(inputs)
                        if torch.isnan(outcome_pred).any():
                            print("NaN detected in outcome_pred")
                            print(outcome_pred)
                        if torch.isnan(outcome).any():
                            print("NaN detected in outcome")
                            print(outcome)
                        loss = cls_criterion(outcome_pred, outcome)
                        total_loss += loss.item() * inputs.size(0)
                        preds = torch.sigmoid(outcome_pred)
                        predicted = (preds > 0.5).float()
                        total_correct += (predicted == outcome).sum().item()
                        total_samples += outcome.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(outcome.cpu().numpy())
                avg_loss = total_loss / len(val_loader.dataset)
                accuracy = total_correct / total_samples
                auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
                print(
                    f"Fold {fold} Val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f}"
                )
                log_dict = {
                    "epoch": epoch + 1,
                    "ValLoss": avg_loss,
                    "Accuracy": accuracy,
                    "AUC": auc,
                }
                metric = accuracy

            with open(log_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow(log_dict)

            if best_metric is None or metric > best_metric:
                best_metric = metric
                checkpoint_path = os.path.join(
                    args.out, f"best_phase{args.phase}_fold{fold}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved for Fold {fold}: {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    args.out = (
        args.out + f"_lr{args.lr}_bs{args.batch_size}_{args.loss}_single_dice_rgb_5fold"
    )
    os.makedirs(args.out, exist_ok=True)
    train(args, torch.device(args.device if torch.cuda.is_available() else "CPU"))
