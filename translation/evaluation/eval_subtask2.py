# pip install git+https://github.com/openai/CLIP.git
# pip install lpips
# pip install dists-pytorch
# pip install scikit-image

import argparse
import os

import clip
import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm

IMG_SIZE = (256, 256)  # resize image to this size for evaluation.
DIR_GT = "ground-truths"
DIR_PRED = "predictions"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir", type=str, required=True, help="The directory of the results"
)
parser.add_argument(
    "--sample_n",
    type=int,
    default=-1,
    help="Randomly sample the number of frames to evaluate. *Use for DEBUG purpose only*",
)


args = parser.parse_args()

res_dir = args.results_dir

dir_gt = os.path.join(args.results_dir, DIR_GT)
dir_pred = os.path.join(args.results_dir, DIR_PRED)

img_names = os.listdir(dir_gt)
print(f"number of images: {len(img_names)}")

if args.sample_n > 0:
    img_names = np.random.choice(img_names, args.sample_n, replace=False)
    print(f"sample {args.sample_n} imgs for evaluation")

ssims = []
psnrs = []
distss = []
lpipss = []
clip_scores = []

device = "cuda" if torch.cuda.is_available() else "cpu"

dists_fn = DISTS().to(device)
lpips_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
)

np.random.shuffle(img_names)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def cal_clip_score(img1: np.ndarray, img2: np.ndarray):
    """calculate clip score.

    Args:
        img1 (np.ndarray): The first image. Shape: [H,W,C]. dtype: uint8.
        img2 (np.ndarray): The second image. Shape: [H,W,C]. dtype: uint8.

    Returns: TODO

    """
    img1 = clip_preprocess(Image.fromarray(img1)).unsqueeze(0).to(device)
    img2 = clip_preprocess(Image.fromarray(img2)).unsqueeze(0).to(device)
    img1_features = clip_model.encode_image(img1)
    img2_features = clip_model.encode_image(img2)
    img1_features = img1_features / img1_features.norm(dim=1, keepdim=True).to(torch.float32)
    img2_features = img2_features / img2_features.norm(dim=1, keepdim=True).to(torch.float32)
    logit_scale = clip_model.logit_scale.exp()
    score = logit_scale * (img1_features * img2_features).sum()
    return score


def load_img(img_path: str):
    """load image to numpy array

    Args:
        img_path (str): path to image.

    Returns: np.ndarray | None. dtype: uint8.
        return None if file not exist or occurred some errors during loading.

    """
    try:
        if not os.path.isfile(img_path):
            print(f"file not existed for image: {img_path}")
            return None

        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize(IMG_SIZE)
        return np.array(img_pil)
    except Exception as e:
        print(f"Exception while loading image: {img_path}: {e}")
        return None


for i, img_name in enumerate(tqdm(img_names)):
    if not img_name.endswith(".png"):
        continue

    img_gt = load_img(os.path.join(dir_gt, img_name))
    if img_gt is None:  # skip if erros with GT image.
        continue

    img_pred = load_img(os.path.join(dir_pred, img_name))

    if img_pred is None:  # set values if missing prediction.
        ssim_value, psnr_value, clip_score, dists_value, lpips_value = 0, 0, 0, 1, 1
    else:
        # SSIM and PSNR
        ssim_value = ssim(img_gt, img_pred, channel_axis=2)
        psnr_value = psnr(img_gt, img_pred)

        with torch.no_grad():
            # clip score
            clip_score = cal_clip_score(img_gt, img_pred).item()

            # DISTS and LPIPS
            img_gt_norm = transform(img_gt).unsqueeze(0).to(device)
            img_pred_norm = transform(img_pred).unsqueeze(0).to(device)
            dists_value = dists_fn(img_gt_norm, img_pred_norm).item()
            lpips_value = lpips_fn(img_gt_norm, img_pred_norm).item()

    ssims.append(ssim_value)
    psnrs.append(psnr_value)
    distss.append(dists_value)
    lpipss.append(lpips_value)
    clip_scores.append(clip_score)

print("ssim={}".format(sum(ssims) / len(ssims)))
print("psnr={}".format(sum(psnrs) / len(psnrs)))
print("dists={}".format(sum(distss) / len(distss)))
print("lpips={}".format(sum(lpipss) / len(lpipss)))
print("clip_score={}".format(sum(clip_scores) / len(clip_scores)))
