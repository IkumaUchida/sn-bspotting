import optuna
from optuna.integration import PyTorchLightningPruningCallback
import tempfile
from tabulate import tabulate
from util.io import load_json, load_gz_json, store_json
from util.eval import smooth_and_peak_detection
from eval import get_pred_file
from eval_ensemble import ensemble
from eval_soccernet_ball import get_args, store_eval_files, load_fps_dict
from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_BALL
from collections import defaultdict
import copy
import os
import glob
import time
import argparse
import shutil



def eval_wrapper(metric, eval_dir, split_name, soccernet_path):
    results = sn_evaluate(
        SoccerNet_path=soccernet_path, Predictions_path=eval_dir,
        split=split_name, version=2, metric=metric, framerate=25, label_files="Labels-ball.json", num_classes=2, dataset="Ball")
    return results["a_mAP"]


def objective(trial,  pred, split, soccernet_path, eval_dir):
    # Suggest values of the hyperparameters
    gaussian_sigma = trial.suggest_float("gaussian_sigma", 1e-5, 1, log=True)
    peak_distance = trial.suggest_int("peak_distance", 1, 50)
    peak_height = trial.suggest_float("peak_height", 0.001, 0.1, step=0.001)

    # Create a unique directory for each trial
    if eval_dir is None:
        eval_dir = "/default/directory"  # replace with your desired default directory

    trial_eval_dir = os.path.join(eval_dir, f"trial_{trial.number}")
    os.makedirs(trial_eval_dir, exist_ok=True)

    new_pred = smooth_and_peak_detection(pred, gaussian_sigma, peak_distance, peak_height)

    store_eval_files(new_pred, trial_eval_dir)
    # print('Done processing prediction files!')
    split_name = split
    if split == 'val':
        split_name = 'valid'
    # Evaluate the results
    score = eval_wrapper('at1', trial_eval_dir, split_name, soccernet_path)

    # Clean up the directory to save disk space
    shutil.rmtree(trial_eval_dir)

    # Return the 'a_map' score to be maximized
    return score


def hpo(pred, split, soccernet_path, eval_dir, n_trials):
    # Define a study and optimize the objective function
    study_name = "hpo_smooth_and_peak_detection"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage=storage_name, load_if_exists=True
    )
    # Pass the additional arguments to the objective function using a lambda function
    study.optimize(lambda trial: objective(trial, pred, split, soccernet_path, eval_dir), n_trials=n_trials)

    print("Best trial:")
    trial_ = study.best_trial
    print(f"  Value: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")


def get_args():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Hyper-parameter optimization for smooth and peak detection in soccer videos.')

    # Adding arguments
    parser.add_argument('--soccernet_path', type=str, default="/workspace/data/soccernet_ball/spotting-ball-2023", help='Path to SoccerNet data.')
    parser.add_argument('--split', type=str, default="test", help='Data split to use for evaluation.')
    parser.add_argument('--pred_file', type=str, default="/workspace/outputs/7/", help='Path to prediction file.')
    parser.add_argument('--n_trials', type=int, default=10, help='trial number of hpo')
    parser.add_argument('--eval_dir', type=str, default=None, help='Path to directory for evaluation files.')
    # Parse arguments
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if len(args.pred_file) == 1:
        pred_file = args.pred_file[0]
        if os.path.isdir(pred_file):
            pred_file, _ = get_pred_file(pred_file, args.split)
            print('Evaluating on: {}'.format(pred_file))
        pred = (load_gz_json if pred_file.endswith('.gz') else load_json)(
                pred_file)
    else:
        scores = []
        fps_dict = None
        for p in glob.glob(args.pred_file):
            if os.path.isdir(p):
                p2, epoch = get_pred_file(p, args.split)
                print('Evaluating on: {}'.format(p))
                if fps_dict is None:
                    fps_dict = load_fps_dict(p2)
                scores.append(load_gz_json(os.path.join(
                    p, 'pred-{}.{}.score.json.gz'.format(args.split, epoch))))
            else:
                if fps_dict is None:
                    fps_dict = load_fps_dict(p.replace('score', 'recall'))
                scores.append(load_gz_json(p))
        _, pred = ensemble('soccernet_ball', scores, fps_dict=fps_dict)

    hpo(pred=pred, split=args.split, soccernet_path=args.soccernet_path, eval_dir=args.eval_dir, n_trials=args.n_trials)

if __name__ == "__main__":
    main()