#!/usr/bin/env python
import os
import argparse
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import NAFNetWithOutcome, UNetWithOutcome3
from sklearn.model_selection import train_test_split


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
        view = row["View"]

        if self.transform:
            pre_img = self.transform(pre_img)
            duration_img = self.transform(duration_img)
            avgforce_img = self.transform(avgforce_img)
            maxtemp_img = self.transform(maxtemp_img)
            maxpower_img = self.transform(maxpower_img)
            post_img = self.transform(post_img)

        input_tensor = torch.cat(
            [pre_img, duration_img, avgforce_img, maxtemp_img, maxpower_img], dim=0
        )
        return input_tensor, post_img, patient_id, view


def save_inference_results(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, targets, pids, view in dataloader:
            inputs = inputs.to(device)
            sim_outputs, _, _ = model(inputs)  # Get simulated post-ablation images

            # Convert tensors to numpy arrays
            inputs_np = inputs.cpu().numpy()
            sim_outputs_np = sim_outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            batch_size = inputs_np.shape[0]
            for i in range(batch_size):
                # Extract components
                pre_img = inputs_np[i, :3].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                duration_img = inputs_np[i, 3]  # (H, W)
                avgforce_img = inputs_np[i, 4]  # (H, W)
                maxtemp_img = inputs_np[i, 5]  # (H, W)
                maxpower_img = inputs_np[i, 6]  # (H, W)
                pred_post = sim_outputs_np[i].transpose(
                    1, 2, 0
                )  # (3, H, W) -> (H, W, 3)
                true_post = targets_np[i].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

                # Clip to [0, 1] range
                pre_img = np.clip(pre_img, 0, 1)
                pred_post = np.clip(pred_post, 0, 1)
                true_post = np.clip(true_post, 0, 1)
                duration_img = np.clip(duration_img, 0, 1)
                avgforce_img = np.clip(avgforce_img, 0, 1)
                maxtemp_img = np.clip(maxtemp_img, 0, 1)
                maxpower_img = np.clip(maxpower_img, 0, 1)

                # Create a grid of images
                fig, axes = plt.subplots(
                    1, 7, figsize=(28, 4)
                )  # 7 columns for Pre, 4 features, Pred Post, True Post
                axes[0].imshow(pre_img)
                axes[1].imshow(duration_img, cmap="gray")
                axes[2].imshow(avgforce_img, cmap="gray")
                axes[3].imshow(maxtemp_img, cmap="gray")
                axes[4].imshow(maxpower_img, cmap="gray")
                axes[6].imshow(true_post)
                axes[5].imshow(pred_post)

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"{pids[i]}_{view[0]}.png")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved inference result to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Phase 1 model")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/dataset_filtered.csv",
        help="Path to CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output_lr0.0001_bs8_l1_single_dice_rgb_v4/epoch_90.pth",
        help="Path to best Phase 1 checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output_90",
        help="Directory to save inference images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Fraction of patients for testing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "CPU")

    # Load data
    df = pd.read_csv(args.csv_path)
    patient_ids = df["ID"].unique()
    # test_ids, _ = train_test_split(
    #     patient_ids, test_size=args.test_split, random_state=42
    # )
    _, test_ids = train_test_split(
        patient_ids, test_size=args.test_split, random_state=42
    )
    test_df = df[df["ID"].isin(test_ids)]
    print(f"Total test patients: {len(test_ids)}")

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    test_dataset = Phase1Dataset(test_df, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Load model (adjust to NAFNetWithOutcome if needed)
    model = UNetWithOutcome3(
        in_channels=7, out_channels=3, num_views=1, freeze_encoder=False
    )
    # If using NAFNetWithOutcome instead:
    # model = NAFNetWithOutcome(in_channels=7, out_channels=3, num_views=1, freeze_encoder=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # Run inference and save results
    save_inference_results(model, test_loader, device, args.output_dir)


if __name__ == "__main__":
    main()
