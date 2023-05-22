import os
import json
import random
import numpy as np

from typing import Dict, List
from collections import defaultdict
from tabulate import tabulate

from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL

import sys
from eval_soccernet_ball import eval_wrapper

def make_json(
    list_of_dict_of_predictions: List[Dict[str, List[List[float]]]],
    eval_dir: str,
    confidence_threshold: float=0.01,
    fps: int= 25):

    """
    This function transforms raw predictions into structured events per game and stores them into corresponding
    directories as JSON files.

    Args:
        list_of_dict_of_predictions (List[Dict[str, List[List[float]]]]): A list of dictionaries where each dictionary has 'video' key is the video path
        and 'predictions' key is a 2D list of model predictions for each frame.
        eval_dir (str): The output directory where the game directories will be created.
        confidence_threshold (float): The minimum confidence required for a prediction to be considered.

    Returns:
        None. The function creates directories and writes JSON files.
    """
    raw_pred = defaultdict(list)
    class_labels = {0: "None", 1: "DRIVE", 2: "PASS"}

    for dict_of_predictions in list_of_dict_of_predictions:

        video = dict_of_predictions['video']
        predictions = np.array(dict_of_predictions['predictions'])

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

    # Create a sub-directory under eval_dir with the game name
    for game, predictions in raw_pred.items():
        game_out_dir = os.path.join(eval_dir, game)

        os.makedirs(game_out_dir, exist_ok=True)
        # Save the dictionary to a json file
        with open(os.path.join(game_out_dir, 'results_spotting.json'), 'w') as outfile:
            json.dump({'UrlLocal': game, 'predictions': predictions}, outfile, indent=4)



def main():
    #Example
    # create your array
    N = 1000

    # Creating predictions for two games
    list_of_dict_of_predictions = []

    games = [
        'england_efl/2019-2020/2019-10-01 - Reading - Fulham',
        'england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town'
    ]

    for game in games:
        for i in range(1, 3):
            prediction = np.array([[random.random(), random.random(), random.random()] for _ in range(N)])
            # normalize each row so that its elements sum to 1
            prediction_normalized = prediction / prediction.sum(axis=1, keepdims=True)

            dict_of_predictions = {}
            dict_of_predictions['video'] = f'{game}/{i}'
            dict_of_predictions['predictions'] = prediction_normalized.tolist()
            list_of_dict_of_predictions.append(dict_of_predictions)

    eval_dir = "./eval_dir/"

    # Call the function
    make_json(list_of_dict_of_predictions, eval_dir, confidence_threshold=0.5)
    soccernet_path = '../data/soccernet_ball'
    split_name = 'test'

    def eval_wrapper(metric):
        results = sn_evaluate(
            SoccerNet_path=soccernet_path, Predictions_path=eval_dir,
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


