import os
import json
import random
import numpy as np
import argparse

from typing import Dict, List
from collections import defaultdict
from tabulate import tabulate

from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL


def make_json(dict_of_predictions: Dict[str, List[List[float]]], output_path: str, confidence_threshold: float = 0.01, fps: int = 25) -> None:
    """
    This function transforms raw predictions into structured events per game and then merges all the predictions for each game into a single JSON file per game.

    Args:
        dict_of_predictions (Dict[str, List[List[float]]]): A dictionary where each key is the video path and value is a 2D list of model predictions for each frame.
        eval_dir (str): The output directory where the game directories will be created.
        confidence_threshold (float, optional): The minimum confidence required for a prediction to be considered. Defaults to 0.01.
        fps (int, optional): Frames per second. Used to calculate the position in the video. Defaults to 25.

    Returns:
        None. The function creates directories and writes JSON files.
    """
    raw_pred = defaultdict(list)
    class_labels = {0: "None", 1: "DRIVE", 2: "PASS"}

    for video, predictions in dict_of_predictions.items():
        predictions = np.array(predictions)

        assert predictions.ndim == 2, "The input array must be 2-dimensional"
        assert predictions.shape[1] == 3, "The input array must have 3 columns"

        # Extract game and half from video
        game, half = video.rsplit('/', 1)
        half = int(half)

        for i, prediction in enumerate(predictions):
            # Get index of max confidence
            max_conf_idx = np.argmax(prediction)

            # Check if confidence is above threshold
            if prediction[max_conf_idx] >= confidence_threshold:
                # Exclude if the label is "None"
                if class_labels[max_conf_idx] != "None":
                    # Calculate position based on frame index
                    ss = i / fps  # assuming fps is 25
                    position = int(ss * 1000)

                    mm = int(ss / 60)
                    ss = int(ss - mm * 60)
                    raw_pred[game].append({
                        'gameTime': '{} - {}:{:02d}'.format(half, mm, ss),
                        'label': class_labels[max_conf_idx],
                        'half': str(half),
                        'position': str(position),
                        'confidence': str(prediction[max_conf_idx])
                    })

        game_out_dir = os.path.join(output_path, game)
        os.makedirs(game_out_dir, exist_ok=True)

        if os.path.exists(os.path.join(game_out_dir, 'results_spotting.json')):
            with open(os.path.join(game_out_dir, 'results_spotting.json'), 'r') as infile:
                data = json.load(infile)
                data['predictions'].extend(raw_pred[game])
        else:
            data = {'UrlLocal': game, 'predictions': raw_pred[game]}

        with open(os.path.join(game_out_dir, 'results_spotting.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)


def main():
    #Example
    parser = argparse.ArgumentParser(description='Transform raw predictions into structured events and save them as JSON.')
    parser.add_argument('--output_path', type=str, default='./eval_dir_/', help='The output directory where the JSON files will be created.')
    args = parser.parse_args([])

    N = 1000
    games = [
        'england_efl/2019-2020/2019-10-01 - Reading - Fulham',
        'england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town'
    ]

    for game in games:
        dict_of_predictions = {}
        for i in range(1, 3):
            prediction = np.array([[random.random(), random.random(), random.random()] for _ in range(N)])
            prediction_normalized = prediction / prediction.sum(axis=1, keepdims=True)
            
            dict_of_predictions[f'{game}/{i}'] = prediction_normalized.tolist()
        
        make_json(dict_of_predictions, args.output_path, confidence_threshold=0.5)


    # Evaluate Example
    soccernet_path = './data/soccernet_ball'
    split_name = 'test'

    def eval_wrapper(metric):
        results = sn_evaluate(
            SoccerNet_path=soccernet_path, Predictions_path=args.output_path,
            split=split_name, version=2, metric=metric, framerate=25, label_files="Labels-ball.json", num_classes=2, dataset="Ball")

        rows = []
        for i in range(len(results['a_mAP_per_class'])):
            label = INVERSE_EVENT_DICTIONARY_BALL[i]
            rows.append((
                label,
                '{:0.2f}'.format(results['a_mAP_per_class'][i] * 100),
                '{:0.2f}'.format(results['a_mAP_per_class_visible'][i] * 100),
                '{:0.2f}'.format(results['a_mAP_per_class_unshown'][i] * 100)
            ))
        rows.append((
            'Average mAP',
            '{:0.2f}'.format(results['a_mAP'] * 100),
            '{:0.2f}'.format(results['a_mAP_visible'] * 100),
            '{:0.2f}'.format(results['a_mAP_unshown'] * 100)
        ))

        print('Metric:', metric)
        print(tabulate(rows, headers=['', 'Any', 'Visible', 'Unseen']))

    eval_wrapper('loose')
    eval_wrapper('tight')
    eval_wrapper('at1')
    eval_wrapper('at2')
    eval_wrapper('at3')
    eval_wrapper('at4')
    eval_wrapper('at5')
    

if __name__ == '__main__':
    main()