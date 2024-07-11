import os
import argparse
import json
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm.auto import tqdm
import pandas as pd

def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim==2:
        return input_img
    
    mask_image = np.zeros(input_img.shape,np.uint8)
    mask_image[:,:,1] = 255
    mask_image = mask_image*np.repeat(binary_mask[:,:,np.newaxis],3,axis=2)

    blend_image = input_img[:,:,:]
    pos_idx = binary_mask>0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:,:,ind]
        ch_img2 = mask_image[:,:,ind]
        ch_img3 = blend_image[:,:,ind]
        ch_img3[pos_idx] = alpha*ch_img1[pos_idx] + (1-alpha)*ch_img2[pos_idx]
        blend_image[:,:,ind] = ch_img3
    return blend_image

def show_img(img):
    plt.figure(facecolor="white", figsize=(30, 10), dpi=100)
    plt.grid("off")
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def save_img(img, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    Image.fromarray(img).save(output)

def main(args):
    with open(args.split_json, 'r') as fp:
        split = json.load(fp)
    
    if split is None:
        print('No split found')
        return
    if args.compute_stats:
        takes = split[args.split]
        df_list = []
        for take_id in tqdm(takes):
            df_i = process_take(take_id[0], args.input, args.output, args.split, args.visualize)
            if df_i is not None:
                df_list.append(df_i)
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(os.path.join(args.output, f'{args.split}.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(args.output, f'{args.split}.csv'))

    empty_mask_portion = len(df[df['mask_portion'] == 0]) / len(df)
    print(f'empty_mask_portion: {empty_mask_portion}')


def process_take(take_id, input, output, split, visualize=False):
    cameras = ['aria01_214-1', 'cam01', 'cam02', 'cam03', 'cam04']
    df_list = []
    for cam in cameras:
        if not os.path.isdir(os.path.join(input, split, take_id, cam)):
            continue
        input_cam_root = os.path.join(input, split, take_id, cam)
        objects = os.listdir(input_cam_root)
        for object_name in objects:
            rgb_root = os.path.join(input_cam_root, object_name, 'rgb')
            rgb_frames = natsorted(os.listdir(rgb_root))
            gt_mask_root = os.path.join(input_cam_root, object_name, 'mask')
            gt_mask_frames = natsorted(os.listdir(gt_mask_root))
            frame_ids = np.intersect1d(rgb_frames, gt_mask_frames)
            for frame_id in frame_ids:
                rgb = np.array(Image.open(os.path.join(rgb_root, frame_id)))
                gt_mask = np.array(Image.open(os.path.join(gt_mask_root, frame_id))) / 255
                mask_portion = np.sum(gt_mask) / (gt_mask.shape[0] * gt_mask.shape[1])

                row = pd.DataFrame([[take_id, cam, object_name, frame_id, mask_portion]], columns=['take', 'camera', 'object', 'frame', 'mask_portion'])
                df_list.append(row)

                if visualize:
                    blended_img = blend_mask(rgb, gt_mask.astype(np.uint8), alpha=0.5)
                    # show_img(blended_img)
                    output_path = os.path.join(output, split, take_id, cam, object_name, 'blend')
                    os.makedirs(output_path, exist_ok=True)
                    save_img(blended_img, os.path.join(output_path, frame_id))

    df = pd.concat(df_list, ignore_index=True) if len(df_list) > 0 else None
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='EgoExo take data root', default='../data/correspondence/dataset')
    parser.add_argument('--split_json', help='EgoExo take data root', default='../data/correspondence/split.json')
    parser.add_argument('--split', help='EgoExo take data root', default='val')
    parser.add_argument('--visualize', action='store_true', help='EgoExo take data root')
    parser.add_argument('--compute_stats', action='store_true', help='EgoExo take data root')
    parser.add_argument('--output', help='Output data root', default='../data/correspondence/dataset')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    main(args)