import os
import argparse
import json
import pandas as pd
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decord import VideoReader
from decord import cpu, cpu
from tqdm.auto import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map

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

def prepare_annotations(annotations, take_id, output):
    take_annotation = annotations.get(take_id)
    scenario = take_annotation['scenario']
    take_name = take_annotation['take_name']
    object_annotations = take_annotation['object_names']['annotation']
    object_masks = take_annotation['object_masks']
    df_list = []
    for object_name, cam_annotations in object_masks.items():
        for cam_name, cam_data in cam_annotations.items():
            cam_annotation = cam_data['annotation']
            for frame_id, annotation in cam_annotation.items():
                # import pdb; pdb.set_trace()
                width = annotation['width']
                height = annotation['height']
                encoded_mask = annotation['encodedMask']
                row = pd.DataFrame([[scenario, take_name, object_name, cam_name, frame_id, width, height, encoded_mask]], 
                                   columns=['scenario', 'take_name', 'object_name', 'cam_name', 'frame_id', 'width', 'height', 'encoded_mask'])
                df_list.append(row)
    df = pd.concat(df_list)
    df.to_csv(output)
    return df

def show_img(img):
    plt.figure(facecolor="white", figsize=(30, 10), dpi=100)
    plt.grid("off")
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def save_img(img, output, new_width=640):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    im = Image.fromarray(img)
    width, height = im.size
    new_size = (new_width, int(height*new_width/width))
    im = im.resize(new_size)
    im.save(output)

def process_take(take_id, input='', output='', split='train'):
    try:
        take_id = take_id[0]
        take_output = os.path.join(output, split, take_id)
        annotation_path = os.path.join(input, take_id, 'annotation.json')
        with open(annotation_path, 'r') as fp:
            annotation = json.load(fp)
        masks = annotation['masks']
        for object_name, cams in masks.items():
            for cam_name, cam_data in cams.items():
                vr = VideoReader(os.path.join(input, take_id, f'{cam_name}.mp4'), ctx=cpu(0))
                print(take_id, object_name, cam_name)
                frame_ids = []
                frame_mask_cocos = []
                count = 0
                for frame_id, frame_mask_coco in cam_data.items():
                    if frame_mask_coco is None:
                        continue
                    if os.path.isfile(os.path.join(take_output, cam_name, object_name, 'mask', f'{frame_id}.png')):
                        continue
                    frame_ids.append(frame_id)
                    frame_mask_cocos.append(frame_mask_coco)
                
                if len(frame_ids)==0:
                    continue
                
                vr_frame_ids = [int(frame_id) // 30 for frame_id in frame_ids]
                frames = vr.get_batch(vr_frame_ids).asnumpy()
                for i, frame_id in enumerate(frame_ids):
                    frame_mask_coco = frame_mask_cocos[i]
                    frame = frames[i]
                    frame_mask = mask_utils.decode(frame_mask_coco)
                    # import pdb; pdb.set_trace()
                    # frame = vr[int(frame_id) // 30].asnumpy()
                    save_img(frame, os.path.join(take_output, cam_name, object_name, 'rgb', f'{frame_id}.png'))
                    save_img(frame_mask*255, os.path.join(take_output, cam_name, object_name, 'mask', f'{frame_id}.png'))
                    # blended_img = blend_mask(frame, frame_mask, alpha=0.5)
                    # save_img(blended_img, os.path.join(take_output, cam_name, object_name, 'blend', f'{frame_id}.png'))
    except Exception as e:
        print(take_id)
        print(e)


def main(args):
    with open(os.path.join(args.input, 'split.json'), 'r') as fp:
        split = json.load(fp)
    
    if split is None:
        print('No split found')
        return
    # import pdb; pdb.set_trace()
    for key, split_i in split.items():
        if key == 'train':
            continue
        # process_map(partial(process_take, input=args.input, output=args.output, split=key), split_i, max_workers=3)
        for take_id in tqdm(split_i):
        #     # import pdb; pdb.set_trace()
            if take_id[0] == args.take:
                process_take(take_id, args.input, args.output, key)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='EgoExo take data root', default='../data/correspondence')
    parser.add_argument('--output', help='Output data root', default='../data/correspondence/dataset')
    parser.add_argument('--take', help='Output data root', default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    main(args)