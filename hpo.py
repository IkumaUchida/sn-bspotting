import optuna
from optuna.integration import PyTorchLightningPruningCallback
import tempfile
from tabulate import tabulate
from util.io import load_json, load_gz_json, store_json
from eval import get_pred_file
from eval_ensemble import ensemble
from eval_soccernet_ball import get_args, store_eval_files, load_fps_dict
from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import copy
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time


def eval_wrapper(metric, eval_dir, split_name):
    results = sn_evaluate(
        SoccerNet_path=soccernet_path, Predictions_path=eval_dir,
        split=split_name, version=2, metric=metric, framerate=25, label_files="Labels-ball.json", num_classes=2, dataset="Ball")
    return results["a_mAP"]

def smooth_and_peak_detection(pred, gaussian_sigma, peak_distance, peak_height, plot_smoothed=False, plot_peaks=False):
    new_pred = []
    for video_pred in pred:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)
        new_events = []
        for label, events in events_by_label.items():
            # Sort events by frame
            events.sort(key=lambda x: x['frame'])
            # Extract scores and apply Gaussian filter
            scores = [e['score'] for e in events]
            smoothed_scores = gaussian_filter1d(scores, sigma=gaussian_sigma)
            # Find peaks
            peaks, _ = find_peaks(smoothed_scores, distance=peak_distance, height=peak_height)
            # Add peak events to new events
            for peak in peaks:
                new_events.append(events[peak])
            # Plotting
            if plot_smoothed:
                plt.figure(figsize=(10, 4))
                plt.plot(scores, label='Original scores')
                plt.plot(smoothed_scores, label='Smoothed scores')
                plt.legend()
                plt.title(f'Video: {video_pred["video"]}, Label: {label}')
                plt.show()
            if plot_peaks:
                plt.figure(figsize=(10, 4))
                plt.plot(smoothed_scores)
                plt.plot(peaks, smoothed_scores[peaks], "x")
                plt.title(f'Video: {video_pred["video"]}, Label: {label} - Peaks')
                plt.show()
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = sorted(new_events, key=lambda x: x['frame'])
        new_video_pred['num_events'] = len(new_events)
        new_pred.append(new_video_pred)
    return new_pred

def objective(trial):
    # Suggest values of the hyperparameters
    gaussian_sigma = trial.suggest_float("gaussian_sigma", 1e-5, 1, log=True)
    peak_distance = trial.suggest_int("peak_distance", 1, 50)
    peak_height = trial.suggest_float("peak_height", 0.001, 0.1, step=0.001)
    # eval_dir = None
    # Create a unique directory for each trial
    eval_dir = f"../outputs/7/eval_dir/trial_{trial.number}"
    os.makedirs(eval_dir, exist_ok=True)
    # print('Using directory for evaluation:', eval_dir)
    # Call the smooth_and_peak_detection function
    new_pred = smooth_and_peak_detection(pred, gaussian_sigma, peak_distance, peak_height)
    store_eval_files(new_pred, eval_dir)
    # print('Done processing prediction files!')
    split_name = split
    if split == 'val':
        split_name = 'valid'
    # Evaluate the results
    score = eval_wrapper('at1', eval_dir, split_name)
    # Clean up the directory to save disk space
    # shutil.rmtree(eval_dir)
    # Return the 'a_map' score to be maximized
    return score

def hpo():
    # Define a study and optimize the objective function
    study_name = "hpo_smooth_and_peak_detection"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage=storage_name, load_if_exists=True
    )
    study.optimize(objective, n_trials=500)
    print("Best trial:")
    trial_ = study.best_trial
    print(f"  Value: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")
# Call the Hyperparameter Optimization function
soccernet_path = "/workspace/data/soccernet_ball/spotting-ball-2023"
split = "test"
pred_file = "/workspace/outputs/7/"
split ="test"
nms_window = 25
if len(pred_file) == 1:
    pred_file = pred_file[0]
    if os.path.isdir(pred_file):
        pred_file, _ = get_pred_file(pred_file, split)
        #assert split != 'challenge', \
        #    'Do not infer pred file if challenge due to bad data!'
        print('Evaluating on: {}'.format(pred_file))
    pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(
            pred_file)
else:
    scores = []
    fps_dict = None
    for p in glob.glob(pred_file):
        if os.path.isdir(p):
            assert split != 'challenge', \
                'Do not infer pred file if challenge due to bad data!'
            p2, epoch = get_pred_file(p, split)
            print('Evaluating on: {}'.format(p))
            if fps_dict is None:
                fps_dict = load_fps_dict(p2)
            scores.append(load_gz_json(os.path.join(
                p, 'pred-{}.{}.score.json.gz'.format(split, epoch))))
        else:
            if fps_dict is None:
                fps_dict = load_fps_dict(p.replace('score', 'recall'))
            scores.append(load_gz_json(p))
    _, pred = ensemble('soccernet_ball', scores, fps_dict=fps_dict)

hpo()