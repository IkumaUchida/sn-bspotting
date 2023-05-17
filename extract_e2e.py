# extract_vector.py
# !/usr/bin/env python3
import os
import argparse
import re
import torch
import numpy as np
from tqdm import tqdm
from dataset.frame import ActionSpotVideoDataset
from util.io import load_json
from util.dataset import load_classes
from train_e2e import E2EModel
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model dir')
    parser.add_argument('frame_dir', help='Path to the frame dir')
    parser.add_argument('output_dir', help='Path to the output dir')
    parser.add_argument('-s', '--split',
                        choices=['train', 'val', 'test', 'challenge'],
                        required=True)
    parser.add_argument('--no_overlap', action='store_true')
    parser.add_argument('-d', '--dataset',
                        help='Dataset name if not inferrable from the config')
    parser.add_argument('--extract', action='store_true', 
                        help='Flag to extract gru vectors or not')
    parser.add_argument('--gpu_parallel', action='store_true',
                        help='Flag to enable multi-GPU parallelism')  # Add gpu_parallel parameter
    return parser.parse_args()


def get_best_epoch(model_dir, key='val_mAP'):
    data = load_json(os.path.join(model_dir, 'loss.json'))
    best = max(data, key=lambda x: x[key])
    return best['epoch']

def get_last_epoch(model_dir):
    regex = re.compile(r'checkpoint_(\d+)\.pt')
    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    assert last_epoch >= 0
    return last_epoch

def extract_gru_features(model_dir, frame_dir, output_dir, split, no_overlap, dataset, extract, gpu_parallel):
    config_path = os.path.join(model_dir, 'config.json')
    config = load_json(config_path)
    if dataset is None:
        dataset = config['dataset']
    classes = load_classes(os.path.join('data', dataset, 'class.txt'))
    if os.path.isfile(os.path.join(model_dir, 'loss.json')):
        best_epoch = get_best_epoch(model_dir)
        print('Best epoch:', best_epoch)
    else:
        best_epoch = get_last_epoch(model_dir)

    model = E2EModel(
        len(classes) + 1, config['feature_arch'], config['temporal_arch'],
        clip_len=config['clip_len'], modality=config['modality'],
        multi_gpu=gpu_parallel, return_states=extract)  # Add return_states parameter

    model.load(torch.load(os.path.join(
        model_dir, 'checkpoint_{:03d}.pt'.format(best_epoch
    ))))
    split_path = os.path.join('data', dataset, '{}.json'.format(split))
    split_data = ActionSpotVideoDataset(
        classes, split_path, frame_dir, config['modality'], config['clip_len'],
        overlap_len=0 if no_overlap else config['clip_len'] // 2,
        crop_dim=config['crop_dim'], skip_partial_end=False, extract_features=extract)

    os.makedirs(model_dir, exist_ok=True)


    with torch.no_grad():
        all_gru_states = []
        prev_video_name = split_data[0]['video']  # Initialize with the name of the first video

        for _, video in enumerate(tqdm(split_data)):
            inputs = video['frame'].to(model.device)
            true_frame_count = video['frame_count']
            # print(video['frame_count'])

            # Check if the video name has changed
            if video['video'] != prev_video_name:
                # Save the GRU states for the previous video
                output_path = os.path.join(output_dir, f'{prev_video_name}.npy')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, np.array(all_gru_states))
                print(f'Saved GRU states for {prev_video_name} to {output_path}')
                # Reset the list of GRU states and update the video name
                all_gru_states = []
                prev_video_name = video['video']

            if extract:
                gru_states_perframe, _ = model.predict_with_gru_states(inputs)  # Get GRU hidden states
                gru_states_perframe = np.squeeze(gru_states_perframe, axis=0)
                # If the number of GRU states is larger than the true frame count, remove the padding states# Remove the first dimension
                if len(gru_states_perframe) > true_frame_count:
                    gru_states_perframe = gru_states_perframe[:true_frame_count]
                all_gru_states.extend(gru_states_perframe)  # Add the GRU states to the list
            else:
                _ = model.predict(inputs)

    # Save the GRU states for the last video
    output_path = os.path.join(output_dir, f'{prev_video_name}.npy')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(all_gru_states))

if __name__ == '__main__':
    extract_gru_features(**vars(get_args()))