from functools import partial
import numpy as np
import json
import optuna
import os, sys
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from collections import defaultdict
import copy
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import argparse
import glob

from itertools import groupby
from operator import itemgetter

# sys.path.append("../")  #
from eval_soccernet_ball import store_eval_files
from SoccerNet.Evaluation.ActionSpotting import evaluate as sn_evaluate
from typing import Dict, List, Tuple, Union


def eval_wrapper(metric, eval_dir, split_name, soccernet_path):
    results = sn_evaluate(
        SoccerNet_path=soccernet_path,
        Predictions_path=eval_dir,
        split=split_name,
        version=2,
        metric=metric,
        framerate=25,
        label_files="Labels-ball.json",
        num_classes=2,
        dataset="Ball",
    )
    return results["a_mAP"]


def threshold_frames_by_area(predictions, arr_1d, threshold):
    # Expects a 1D array containing the area of the largest bounding box in each frame
    arr_1d = np.array(arr_1d)
    assert arr_1d.ndim == 1, "arr_1d must be 1D"

    thresh_frames = np.where(arr_1d > threshold, 0, 1)
    filtered_predictions = predictions * thresh_frames[:, None]

    return filtered_predictions


def make_json(
    dict_of_predictions: Dict[str, np.ndarray],
    output_path: str,
    confidence_threshold: float = 0.01,
    fps: int = 25,
) -> None:
    """
    This function transforms raw predictions into structured events per game and then merges all the predictions for each game into a single JSON file per game.

    Args:
        dict_of_predictions (Dict[str, np.ndarray]): A dictionary where each key is the video path and value is a 2D numpy array of model predictions for each frame.
        output_path (str): The output directory where the game directories will be created.
        confidence_threshold (float, optional): The minimum confidence required for a prediction to be considered. Defaults to 0.01.
        fps (int, optional): Frames per second. Used to calculate the position in the video. Defaults to 25.

    Returns:
        None. The function creates directories and writes JSON files.
    """
    for video, predictions in dict_of_predictions.items():
    
        game, half = video.rsplit("/", 1)
        game_out_dir = os.path.join(
            output_path, os.path.join(*game.split(os.path.sep)[-3:])
        )
        save_path = os.path.join(game_out_dir, "results_spotting.json")
        # delete save_path if it exists
        if os.path.exists(save_path):
            os.remove(save_path)

    raw_pred = defaultdict(list)
    class_labels = {0: "None", 1: "DRIVE", 2: "PASS"}

    for video, predictions in dict_of_predictions.items():
        assert isinstance(predictions, np.ndarray), "The input must be a numpy array"
        assert predictions.ndim == 2, "The input array must be 2-dimensional"
        assert predictions.shape[1] == 3, "The input array must have 3 columns"

        game, half = video.rsplit("/", 1)
        half = int(half)
        raw_pred = defaultdict(list)

        for i, prediction in enumerate(predictions):
            max_conf_idx = np.argmax(prediction)

            if prediction[max_conf_idx] >= confidence_threshold and class_labels[max_conf_idx] != "None":
                ss = i / fps
                position = int(ss * 1000)

                mm = int(ss / 60)
                ss = int(ss - mm * 60)

                raw_pred[game].append(
                    {
                        "gameTime": "{} - {}:{:02d}".format(half, mm, ss),
                        "label": class_labels[max_conf_idx],
                        "half": str(half),
                        "position": str(position),
                        "confidence": str(prediction[max_conf_idx]),
                    }
                )

        game_out_dir = os.path.join(
            output_path, os.path.join(*game.split(os.path.sep)[-3:])
        )
        os.makedirs(game_out_dir, exist_ok=True)

        save_path = os.path.join(game_out_dir, "results_spotting.json")

        if os.path.exists(save_path):
            with open(save_path, "r") as infile:
                data = json.load(infile)
                data["predictions"].extend(raw_pred[game])
        else:
            data = {"UrlLocal": game, "predictions": raw_pred[game]}
        
        # Remove duplicates
        data["predictions"] = [dict(t) for t in set(tuple(d.items()) for d in data["predictions"])]

        # Sort by "half" and then "position"
        data["predictions"].sort(key=lambda x: (int(x['half']), int(x['position'])))

        with open(save_path, "w") as outfile:
            json.dump(data, outfile, indent=4)
        print(f"Wrote predictions to '{save_path}'")



def post_process(
    prediction_results,
    sg_window_length,
    sg_polyorder,
    peak_distance,
    peak_height,
    nms,
    bbox_threshold=0.0,
    ignore_bbox=False
):
    """Performs smoothing, peak detection, and thresholding on the predictions

    Returns a dictionary where each key is a
    """
    output = {}
    for filepath in prediction_results:
        # Load predictions
        predictions = np.load(filepath)
        assert predictions.shape[1] == 3

        smooth_predictions = np.zeros_like(predictions)
        filtered_predictions = np.zeros_like(predictions)

        # iterate by label
        for i in range(1, 3):
            smooth_scores = savgol_filter(
                predictions[:, i], sg_window_length, sg_polyorder
            )

            smooth_scores = maximum_filter1d(smooth_scores, size=nms)
            smooth_predictions[:, i] = smooth_scores

            # Find peaks
            peaks, _ = find_peaks(
                smooth_scores, distance=peak_distance, height=peak_height
            )

            # Add peaks to filtered_predictions
            filtered_predictions[peaks, i] = predictions[peaks, i]

        # Load bbox_areas and apply thresholding only if ignore_bbox is False
        if not ignore_bbox and bbox_threshold > 0:
            bbox_areas = np.load(f"{filepath[:-4]}_bbox_areas.npy")[: len(predictions)]

            # Apply thresholding
            filtered_predictions = threshold_frames_by_area(
                filtered_predictions, bbox_areas, threshold=bbox_threshold
            )

        # import matplotlib.pyplot as plt

        # # Create a figure with three subplots (one for each class)
        # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        # # Define class labels
        # class_labels = ["None", "Drive", "Pass"]

        # # Iterate by label
        # for i in range(3):
        #     # Plot smoothed scores
        #     axs[i].plot(smooth_predictions[:, i], label="Smoothed Scores")

        #     # Highlight the peaks in the predictions
        #     axs[i].plot(peaks, smooth_predictions[peaks, i], "x", label="Peaks")

        #     # Set title and labels
        #     axs[i].set_title(class_labels[i])
        #     axs[i].set_ylabel("Score")

        #     # Add legend
        #     axs[i].legend(loc="upper right")

        # # Set common x-label
        # axs[2].set_xlabel("Frames")

        # # Save the figure
        # plt.tight_layout()
        # plt.savefig(f"{filepath[:-4]}_predictions_plot.png")

        video_key = os.path.splitext(filepath)[0]

        # Store filtered predictions in the output dictionary
        output[video_key] = filtered_predictions
        # print(video_key)
        # # save output as file
        # with open('./out.json', "w") as outfile:
        #     json.dump(output, outfile)
        # print(f"Wrote predictions to out")           

    return output


def objective(trial, args):
    # Suggest values of the hyperparameters
    sg_window_length = trial.suggest_int("sg_window_length", 3, 99, step=2)
    sg_polyorder = trial.suggest_int("sg_polyorder", 1, min(10, sg_window_length - 1))
    peak_distance = trial.suggest_int("peak_distance", 1, 50)
    peak_height = trial.suggest_float("peak_height", 0.001, 0.1, step=0.001)
    nms = trial.suggest_int("nms", 1, 100)

    if args.ignore_bbox:
        print("Ignoring bounding box thresholding.")
        bbox_threshold = 0
    else:
        bbox_threshold = trial.suggest_float("bbox_threshold", 0, 100)

    # prediction_results is a list containing the names of result .npy files
    # bounding box areas shouhld be stored in a list of .npy files with the same name as the prediction results, but with _bbox_areas appended right before the .npy extension
    # e.g. pred.npy and pred_bbox_areas.npy
    args.sg_window_length = sg_window_length
    args.sg_polyorder = sg_polyorder
    args.peak_distance = peak_distance
    args.peak_height = peak_height
    args.nms = nms
    args.bbox_threshold = bbox_threshold

    score = run(args)
    return score


def run(args):
    pred_files = args.pred_files
    sg_window_length = args.sg_window_length
    sg_polyorder = args.sg_polyorder
    peak_distance = args.peak_distance
    peak_height = args.peak_height
    bbox_threshold = args.bbox_threshold
    nms = args.nms
    ignore_bbox = args.ignore_bbox

    if args.run:
        print(f"parameters")
        print(f"\tsg_window_length: {sg_window_length}")
        print(f"\tsg_polyorder: {sg_polyorder}")
        print(f"\tpeak_distance: {peak_distance}")
        print(f"\tpeak_height: {peak_height}")
        print(f"\tnms: {nms}")
        # print(f"\tbbox_threshold: {bbox_threshold}")

    dict_of_predictions = post_process(
        pred_files,
        sg_window_length,
        sg_polyorder,
        peak_distance,
        peak_height,
        nms,
        bbox_threshold,
        ignore_bbox,
    )

    # Create a unique directory for each trial
    if args.output_dir is None:
        with TemporaryDirectory() as tmpdir:
            make_json(dict_of_predictions, output_path=tmpdir)
            score = eval_wrapper("at1", tmpdir, args.split, args.soccernet_path)
    else:
        make_json(dict_of_predictions, output_path=args.output_dir)
        score = eval_wrapper("at1", args.output_dir, args.split, args.soccernet_path)

    return score


def hpo(args):
    # Define a study and optimize the objective function
    study_name = "hpo_smooth_and_peak_detection"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    # Define args used in the objective function
    split = args.split
    soccernet_path = args.soccernet_path

    # add args to the objective
    objective_with_args = partial(objective, args=args)
    # Create a study if it does not exist
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective_with_args, n_trials=args.n_trials)

    # Print the best trial
    print("Best trial:")
    trial_ = study.best_trial
    print(f"  Value: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")


def get_args():
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Hyper-parameter optimization for smooth and peak detection in soccer videos."
    )

    # Adding arguments
    parser.add_argument(
        "--soccernet_path",
        type=str,
        default="/workspace/data/soccernet_ball/spotting-ball-2023",
        help="Path to SoccerNet data.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Data split to use for evaluation."
    )
    parser.add_argument(
        "--pred_files", nargs="+", type=str, help="Path(s) to prediction file(s)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to directory for output files.",
    )
    parser.add_argument("--n_trials", type=int, default=10, help="trial number of hpo")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Path to directory for evaluation files.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the model with the best hyper-parameters.",
    )
    parser.add_argument(
        "--hpo", action="store_true", help="Run hyper-parameter optimization."
    )
    parser.add_argument(
        "--sim", action="store_true", help="Simulate predictions and bbox_areas."
    )
    parser.add_argument(
        "--ignore_bbox",
        action="store_true",
        help="Flag to ignore bounding box thresholding in post-processing",
    )

    # add other arguments as needed
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=0.175,
        help="Value for gaussian_sigma. Default: 1.0",
    )
    parser.add_argument(
        "--sg_window_length",
        type=int,
        default=51,
        help="Value for sg_window_length. Default: 51",
    )
    parser.add_argument(
        "--sg_polyorder",
        type=int,
        default=3,
        help="Value for sg_polyorder. Default: 3",
    )
    parser.add_argument(
        "--peak_distance",
        type=int,
        default=14,
        help="Value for peak_distance. Default: 10",
    )
    parser.add_argument(
        "--peak_height",
        type=float,
        default=0.029,
        help="Value for peak_height. Default: 20",
    )
    parser.add_argument(
        "--nms",
        type=int,
        default=1,
        help="Value for nms. Default: 1",
    )
    parser.add_argument(
        "--bbox_threshold",
        type=float,
        default=0.0,
        help="Value for bbox_threshold. Default: 0.5",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if os.path.isdir(args.pred_files[0]):
        # glob all files in the directory recursively
        args.pred_files = list(
            glob.iglob(os.path.join(args.pred_files[0], "**/[0-9].npy"), recursive=True)
        )
    if len(args.pred_files) == 0:
        raise ValueError("No prediction files found.")

    if args.run:
        score = run(args)
        print(score)
    if args.hpo:
        hpo(args)
    if args.sim:
        # Simulate predictions, 2d-array of shape (7500, 3)
        predictions = np.random.random((7500, 3))

        # Simulate bbox_areas, 1d-array of shape 7500
        bbox_areas = np.random.randint(1, 100, size=7500)
        thresh_frames = threshold_frames_by_area(bbox_areas, threshold=50)

        with TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "1.npy"), predictions)
            np.save(os.path.join(tmpdir, "1_bbox_areas.npy"), bbox_areas)

            args.pred_files = [os.path.join(tmpdir, "1.npy")]
            output = run(args)
            print("Finished simulating predictions and bbox_areas.")
            print(output)

# python post_processing.py --run --split test --pred_files ./outputs/predictions/test --soccernet_path ./data/soccernet_ball/spotting-ball-2023/ --ignore_bbox --output_dir outputs/ensemble/test --nms 7 --peak_distance 16 --peak_height 0.027 --sg_polyorder 9 --sg_window_length 33
# python post_processing.py --hpo --split train --pred_files ./outputs/predictions/train --soccernet_path ./data/soccernet_ball/spotting-ball-2023/ --n_trials 300 --ignore_bbox
