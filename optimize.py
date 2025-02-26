#!/usr/bin/env python
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import SOFANet
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF  


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
        inputs, targets, masks_abl = [], [], []
        for _, row in data.iterrows():
            pre_img = Image.open(row["Pre"]).convert("RGB")
            duration_img = Image.open(row["Pre_DurationTime"]).convert("L")
            avgforce_img = Image.open(row["Pre_AverageForce"]).convert("L")
            maxtemp_img = Image.open(row["Pre_MaxTemperature"]).convert("L")
            maxpower_img = Image.open(row["Pre_MaxPower"]).convert("L")
            post_img = Image.open(row["Post"]).convert("RGB")
            mask_abl_img = Image.open(row["MaskAblClosing"]).convert("L")
            if self.transform:
                pre_img = self.transform(pre_img)
                duration_img = self.transform(duration_img)
                avgforce_img = self.transform(avgforce_img)
                maxtemp_img = self.transform(maxtemp_img)
                maxpower_img = self.transform(maxpower_img)
                post_img = self.transform(post_img)
                mask_abl_img = self.transform(mask_abl_img)

            input_tensor = torch.cat(
                [pre_img, duration_img, avgforce_img, maxtemp_img, maxpower_img], dim=0
            )
            inputs.append(input_tensor)
            targets.append(post_img)
            masks_abl.append(mask_abl_img)
        inputs = torch.stack(inputs)  # shape: (num_views, 7, H, W)
        targets = torch.stack(targets)  # shape: (num_views, 3, H, W)
        masks_abl = torch.stack(masks_abl)  # shape: (num_views, 1, H, W)
        outcome = float(data["PrimaryAAOutcome"].iloc[0])
        outcome = torch.tensor(outcome, dtype=torch.float)
        return inputs, targets, masks_abl, patient_id, outcome


def optimize_ablation_parameters(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SOFANet(out_channels=3, num_views=args.num_views, freeze_encoder=True)
    checkpoint = torch.load(args.phase2_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    df = pd.read_csv(args.csv_path)
    dataset = Phase2Dataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    param_bounds = {
        "duration": (0.0, 1.0),
        "force": (0.0, 1.0),
        "temp": (0.0, 1.0),
        "power": (0.0, 1.0),
    }

    all_results = []
    lambda_reg = args.lambda_reg

    def outcome_loss(logits):
        return -torch.log(1 - torch.sigmoid(logits) + 1e-8).mean()

    for batch_idx, (inputs, _, masks_abl, pids, outcome) in enumerate(loader):
        pre_images = inputs[0, :, :3, :, :].to(device)  # Shape: (num_views, 3, H, W)
        original_params = (
            inputs[0, :, 3:7, :, :].clone().to(device)
        )  # Shape: (num_views, 4, H, W)
        masks_abl = masks_abl.to(device).squeeze(0)  # Shape: (num_views, 1, H, W)

        # Binarize ablation masks, keep as boolean
        masks_abl = masks_abl > 0.5  # Shape: (num_views, 1, H, W), dtype=torch.bool

        adv_params = original_params.clone().detach()
        adv_params.requires_grad = True

        optimizer = optim.Adam([adv_params], lr=args.lr)

        for step in range(args.max_steps):
            optimizer.zero_grad()
            full_input_views = torch.cat([pre_images, adv_params], dim=1)
            full_input = full_input_views.unsqueeze(0)
            _, outcome_pred, _ = model(full_input)
            outcome_pred_mean = outcome_pred.mean()

            loss_outcome = outcome_loss(outcome_pred_mean)
            loss_reg = F.mse_loss(adv_params, original_params)
            loss = loss_outcome + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                masks_abl_expanded = masks_abl.expand_as(adv_params)
                adv_params[:] = torch.where(
                    masks_abl_expanded, adv_params, original_params
                )
                for i, key in enumerate(["duration", "force", "temp", "power"]):
                    adv_params[:, i] = torch.clamp(
                        adv_params[:, i], param_bounds[key][0], param_bounds[key][1]
                    )

                for i in range(adv_params.shape[1]):
                    adv_params[:, i] = TF.gaussian_blur(
                        adv_params[:, i].unsqueeze(1),
                        kernel_size=args.gaussian_kernel_size,
                        sigma=args.gaussian_sigma,
                    ).squeeze(1)

            if (step + 1) % 10 == 0:
                print(
                    f"Patient {pids}: Step {step+1}/{args.max_steps}, Loss: {loss.item():.6f}, Outcome: {torch.sigmoid(outcome_pred_mean).item():.6f}"
                )

        with torch.no_grad():
            full_input_orig = torch.cat([pre_images, original_params], dim=1).unsqueeze(
                0
            )
            full_input_opt = torch.cat([pre_images, adv_params], dim=1).unsqueeze(0)
            sim_outputs_orig, outcome_pred_orig, _ = model(full_input_orig)
            sim_outputs_opt, outcome_pred_opt, _ = model(full_input_opt)

            sim_orig = sim_outputs_orig.squeeze(0)  # Shape: (num_views, 3, H, W)
            sim_opt = sim_outputs_opt.squeeze(0)  # Shape: (num_views, 3, H, W)

            original_outcomes_per_view = torch.sigmoid(outcome_pred_orig).cpu().numpy()
            optimized_outcomes_per_view = torch.sigmoid(outcome_pred_opt).cpu().numpy()
            avg_original_outcome = original_outcomes_per_view.mean()
            avg_optimized_outcome = optimized_outcomes_per_view.mean()

        result = {
            "ID": pids,
            "original_params": original_params.cpu().numpy(),
            "optimized_params": adv_params.detach().cpu().numpy(),
            "original_outcomes_per_view": original_outcomes_per_view.tolist(),
            "optimized_outcomes_per_view": optimized_outcomes_per_view.tolist(),
            "avg_original_outcome": float(avg_original_outcome),
            "avg_optimized_outcome": float(avg_optimized_outcome),
        }
        all_results.append(result)

        num_views = pre_images.shape[0]
        channel_names = ["DurationTime", "AverageForce", "MaxTemperature", "MaxPower"]

        for view in range(num_views):
            # Visualization 1: Original and Optimized Features
            fig, axs = plt.subplots(
                3, 4, figsize=(16, 12)
            )  # 3 rows: Original, Optimized, Difference
            for i in range(4):
                orig_channel = original_params[view, i].detach().cpu().numpy()
                opt_channel = adv_params[view, i].detach().cpu().numpy()
                diff_channel = orig_channel - opt_channel  # Difference map
                mask_abl_view = masks_abl[view].float().squeeze(0).cpu().numpy()

                # Original
                axs[0, i].imshow(orig_channel, cmap="gray")
                axs[0, i].set_title(
                    f"Original {channel_names[i]}", fontsize=16
                )  # Increased font size
                axs[0, i].axis("off")

                # Optimized
                axs[1, i].imshow(opt_channel, cmap="gray")
                axs[1, i].set_title(
                    f"Optimized {channel_names[i]}", fontsize=16
                )  # Increased font size
                axs[1, i].axis("off")

                # Difference with masked colormap (red for positive, blue for negative)
                diff_masked = np.zeros_like(diff_channel)
                diff_masked[mask_abl_view > 0.5] = diff_channel[
                    mask_abl_view > 0.5
                ]  # Apply mask
                im = axs[2, i].imshow(
                    diff_masked, cmap="RdBu", vmin=-1, vmax=1
                )  # Red-blue colormap
                axs[2, i].set_title(
                    f"Difference {channel_names[i]}", fontsize=16
                )  # Increased font size
                axs[2, i].axis("off")
                plt.colorbar(im, ax=axs[2, i], fraction=0.046, pad=0.04)

            viz_path = os.path.join(
                args.output_dir, f"patient_{pids[0]}_view_{view}_features.png"
            )
            plt.tight_layout()
            plt.savefig(viz_path)
            plt.close()

    # Compute and save overall averages across all patients
    avg_original_all_patients = np.mean(
        [r["avg_original_outcome"] for r in all_results]
    )
    avg_optimized_all_patients = np.mean(
        [r["avg_optimized_outcome"] for r in all_results]
    )
    print(
        f"Average Original Outcome Across All Patients: {avg_original_all_patients:.3f}"
    )
    print(
        f"Average Optimized Outcome Across All Patients: {avg_optimized_all_patients:.3f}"
    )

    # Save averages to a .txt file
    averages_file = os.path.join(args.output_dir, "outcome_averages.txt")
    with open(averages_file, "w") as f:
        f.write(
            f"Average Original Outcome Across All Patients: {avg_original_all_patients:.3f}\n"
        )
        f.write(
            f"Average Optimized Outcome Across All Patients: {avg_optimized_all_patients:.3f}\n"
        )
    print(f"Averages saved to {averages_file}")

    torch.save(all_results, os.path.join(args.output_dir, "optimized_params.pt"))
    print("Optimization results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Optimize Ablative Features")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/dataset_filtered.csv",
        help="Path to CSV file with data information",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (use 1 for patient-wise optimization)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for ablative feature optimization",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=6,
        help="Number of views per patient",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_lr0.0001_bs8_l1_single_dice_rgb_v3",
        help="Directory to save optimized parameters and visualizations",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Regularization weight for preserving original ablative features",
    )
    parser.add_argument(
        "--gaussian_kernel_size",
        type=int,
        default=5,
        help="Kernel size for Gaussian blur (must be odd)",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian blur",
    )
    args = parser.parse_args()
    args.phase2_checkpoint = os.path.join(args.output_dir, "best_phase2.pth")
    # Update output_dir with Gaussian parameters
    args.output_dir = os.path.join(
        args.output_dir,
        f"phase3_abl_closing_ks{args.gaussian_kernel_size}_sigma{args.gaussian_sigma}",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    optimize_ablation_parameters(args)
