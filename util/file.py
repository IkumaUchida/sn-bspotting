import os
import json
import random
import numpy as np

from typing import Dict, List
from collections import defaultdict
from tabulate import tabulate

from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL

def make_json(
    dict_of_predictions: Dict[str, List[List[float]]],
    confidence_threshold: float=0.01,
    fps: int= 25) -> Dict[str, List[Dict[str, str]]]:
    """
    This function processes the raw predictions into structured events per half game.

    Args:
        dict_of_predictions (Dict[str, List[List[float]]]): A dictionary where 'video' key is the video path
        and 'predictions' key is a 2D list of model predictions for each frame.
        confidence_threshold (float, optional): The minimum confidence required for a prediction to be considered. Defaults to 0.01.
        fps (int, optional): Frame rate of the video. Defaults to 25.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary where the key is the game name and the value is a list of
        predictions for each event in the game.
    """

    raw_pred = defaultdict(list)
    class_labels = {0: "None", 1: "DRIVE", 2: "PASS"}

    video = dict_of_predictions['video']
    predictions = np.array(dict_of_predictions['predictions'])

    assert predictions.ndim == 2, "The input array must be 2-dimensional"
    assert predictions.shape[1] == 3, "The input array must have 3 columns"

    game, half = video.rsplit('/', 1)
    half = int(half)

    for i, prediction in enumerate(predictions):
        max_conf_idx = np.argmax(prediction)

        if prediction[max_conf_idx] >= confidence_threshold:
            if class_labels[max_conf_idx] != "None":
                ss = i / fps
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

    return raw_pred


def merge_json(
    eval_dir: str, 
    list_of_game_dicts: List[Dict[str, List[Dict[str, str]]]]) -> None:
    """
    This function merges all the predictions for each game into a single JSON file per game.

    Args:
        eval_dir (str): The output directory where the game directories will be created.
        list_of_game_dicts (List[Dict[str, List[Dict[str, str]]]]): A list of dictionaries where each dictionary
        contains the predictions for a game.

    Returns:
        None. The function creates directories and writes JSON files.
    """
    for game_dict in list_of_game_dicts:
        game_out_dir = os.path.join(eval_dir, list(game_dict.keys())[0])
        os.makedirs(game_out_dir, exist_ok=True)

        if os.path.exists(os.path.join(game_out_dir, 'results_spotting.json')):
            with open(os.path.join(game_out_dir, 'results_spotting.json'), 'r') as infile:
                data = json.load(infile)
                data['predictions'].extend(list(game_dict.values())[0])
        else:
            data = {'UrlLocal': list(game_dict.keys())[0], 'predictions': list(game_dict.values())[0]}

        with open(os.path.join(game_out_dir, 'results_spotting.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)


def main():
    #example
    N = 1000
    games = [
        'england_efl/2019-2020/2019-10-01 - Reading - Fulham',
        'england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town'
    ]
    list_of_game_dicts = []

    for game in games:
        for i in range(1, 3):
            prediction = np.array([[random.random(), random.random(), random.random()] for _ in range(N)])
            prediction_normalized = prediction / prediction.sum(axis=1, keepdims=True)

            dict_of_predictions = {}
            dict_of_predictions['video'] = f'{game}/{i}'
            dict_of_predictions['predictions'] = prediction_normalized.tolist()
            
            game_dict = make_json(dict_of_predictions, confidence_threshold=0.5)
            list_of_game_dicts.append(game_dict)
    eval_dir = './eval_dir/'

    # Call the function
    merge_json(eval_dir, list_of_game_dicts)
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

if __name__ == '__main__':
    main()

