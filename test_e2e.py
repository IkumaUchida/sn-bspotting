#!/usr/bin/env python3
""" Inference for E2E-Spot """

import os
import argparse
import re
import torch
import numpy as np
import json

from dataset.frame import ActionSpotVideoDataset, UnlabeledVideoDataset
from util.io import load_json
from util.dataset import load_classes
from train_e2e import E2EModel, evaluate, inference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model dir')
    parser.add_argument('frame_dir', help='Path to the frame dir')
    parser.add_argument('-s', '--split',
                        choices=['train', 'val', 'test', 'challenge'],
                        required=True)
    parser.add_argument('--no_overlap', action='store_true')
    parser.add_argument('--evaluate_flag', action='store_true', help='Evaluate the model')
    parser.add_argument('--inference_flag', action='store_true', help='Predict using the model')


    save = parser.add_mutually_exclusive_group()
    save.add_argument('--save', action='store_true',
                      help='Save predictions with default names')
    save.add_argument('--save_as', help='Save predictions with a custom prefix')

    parser.add_argument('-d', '--dataset',
                        help='Dataset name if not inferrable from the config')
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

def predict(model, data):
    results = []
    for frame in data:
        pred_cls, pred_prob = model.predict(frame)
        results.append(pred_prob)
    return np.array(results)


def main(model_dir, frame_dir, split, no_overlap, save, save_as, evaluate_flag, inference_flag, dataset):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as fp:
        print(fp.read())

    config = load_json(config_path)
    if os.path.isfile(os.path.join(model_dir, 'loss.json')):
        best_epoch = get_best_epoch(model_dir)
        print('Best epoch:', best_epoch)
    else:
        best_epoch = get_last_epoch(model_dir)

    if dataset is None:
        dataset = config['dataset']
    else:
        if dataset != config['dataset']:
            print('Dataset mismatch: {} != {}'.format(
                dataset, config['dataset']))

    classes = load_classes(os.path.join('../../../', 'data', dataset, 'class.txt'))

    model = E2EModel(
        len(classes) + 1, config['feature_arch'], config['temporal_arch'],
        clip_len=config['clip_len'], modality=config['modality'],
        multi_gpu=config['gpu_parallel'])
    model.load(torch.load(os.path.join(
        model_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

    pred_file = None
    if save_as is not None:
        pred_file = save_as
    elif save is not None:
        pred_file = os.path.join(
            model_dir, 'pred-{}.{}'.format(split, best_epoch))
        assert not os.path.exists(pred_file), pred_file

    if pred_file is not None:
        print('Saving predictions:', pred_file)

    if evaluate_flag:
        split_path = os.path.join('../../../', 'data', dataset, '{}.json'.format(split))
        split_data = ActionSpotVideoDataset(
            classes, split_path, frame_dir, config['modality'], config['clip_len'],
            overlap_len=0 if no_overlap else config['clip_len'] // 2,
            crop_dim=config['crop_dim'])
        
        # err, f1, pred_events, pred_events_high_recall, pred_scores, avg_mAP = evaluate(model, split_data, split.upper(), classes, pred_file,
        #         calc_stats=False)
        
        # print("pred_events: ", pred_events_high_recall)
        # print("type(pred_events): ", type(pred_events_high_recall))
        # #save pred_scores as json
        # with open('output_bas.json', 'w') as f:
        #     json.dump(pred_scores, f)

    if inference_flag:
        print("config['clip_len']: ", config['clip_len'])
        split_data = UnlabeledVideoDataset(
            frame_dir, 
            modality=config['modality'], 
            clip_len=config['clip_len'],
            overlap_len=0 if no_overlap else config['clip_len'] // 2,
            crop_dim=config['crop_dim'])
        
        #check getitem
        print("Length of split_data: ", len(split_data))
        # print(split_data[1])

        pred_events, pred_events_high_recall, pred_scores = inference(model, split_data, classes)
        
        print("pred_events_high_recall: ", pred_events_high_recall)
        print("type(pred_events_high_recall): ", type(pred_events_high_recall))
        #save pred_scores as json
        with open('output.json_224p', 'w') as f:
            json.dump(pred_scores, f)
def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

if __name__ == '__main__':
    main(**vars(get_args()))