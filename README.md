# **Baseline approach for ball action spotting**

# Overview

This repository is the project page for the SoccerNet Challenge's Ball Action Spotting competition.

E2E-spot is used as the baseline method. ([paper](https://arxiv.org/pdf/2207.10213.pdf), [code](https://github.com/jhong93/spot))




# [4/6 updated] Check the baseline operation
---

You can implement the following flow from learning to evaluation.

# Environment
The code is tested in Linux (Ubuntu 16.04 and 20.04) with the dependency versions in `requirements.txt`. Similar environments are likely to work also but YMMV.

Note: If you want to build an environment with Docker and Poetryni, it is preferable to refer to [cvpaperchallenge/Ascender repository](https://github.com/cvpaperchallenge/Ascender).

# usage

## Setting up SoccerNet Ball Action Spotting

Set up SoccerNet for the spotting challenge. For more information about the task, refer to https://www.soccer-net.org/.

1. Follow the instructions on the [SoccerNet github repository](https://github.com/SoccerNet/sn-spotting#soccernet-ball-action-spotting) to obtain the videos and the labels for the action spotting task. Either the 224P or the 720P videos work. The downloaded data should preferably be placed in the `data` directory of this repository.

2. Install SoccerNet dependencies: `pip3 install SoccerNet`. Other packages such as `moviepy` may also be required().

3. Extract frames at 25 FPS using `python3 frames_as_jpg_soccernet_ball.py <video_dir> -o <out_dir> --sample_fps 25`. This will take a while.

* `<video_dir>`: path to the downloaded videos
* `<out_dir>`: path to save the extracted frames

4. Parse the labels with `python3 parse_soccernet_ball.py <label_dir> <frame_dir> -o data/soccernet_ball`.
* `<label_dir>`: path to the downloaded labels(same path as video_dir.)
* `<frame_dir>`: path to the extracted frames(same path as out_dir.)


## Training and evaluation

To train a model, use `python3 train_e2e.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch>`.

* `<dataset_name>`: just select `"soccernet_ball"` 
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., RegNet-Y 200MF w/ GSM : `rny002_gsm`)

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

To evaluate a set of predictions with the mean-AP metric, use `python3 eval.py -s <split> <model_dir_or_prediction_file>`.
* `<model_dir_or_prediction_file>`: can be the saved directory of a model containing predictions or a path to a prediction file.

The predictions are saved as either `pred-{split}.{epoch}.recall.json.gz` or `pred-{split}.{epoch}.json` files. The latter contains only the top class predictions for each frame, omitting all background, while the former contains all non-background detections above a low threshold, to complete the precision-recall curve.

## upload to the evaluation server

To create a SoccerNet package for evaluation, use `eval_soccernet_ball.py`

`python eval_soccernet_ball.py <out_dir> -l <video_dir> --eval_dir <out_dir_pred> -s test --nms_window=25`.
`python eval_soccernet_ball.py <out_dir> -l <video_dir> --eval_dir <out_dir_pred> -s challenge --nms_window=25`.

* `<out_dir>`: path to the saved directory of a model containing predictions
* `<video_dir>`: path to the downloaded videos
* `<out_dir_pred>`: path to save the predictions for evaluation

To submit to the evaluation server, simply zip all files inside `<out_dir_pred>`

### Prediction file format in <out_dir_pred>

Predictions are formatted similarly to the labels:
```
[
    {
        "video": VIDEO_ID,
        "events": [
            {
                "frame": 525,               // Frame
                "label": CLASS_NAME,        // Event class
                "score": 0.96142578125
            },
            ...
        ],
        "fps": 25           // Metadata about the source video
    },
    ...
]
```

## Visualization
To visualize the predictions, use `python3 Visualization/visualize.py --input_video <input_video_file> --input_json <input_json_file> --output_video_dir <output_video_dir> --output_video_name <output_video_name> --start <start_time> --end <end_time>` from root dir.

* `<input_video_file>`: path to the input video file
* `<input_json_file>`: path to the input json file
* `<output_video_dir>`: path to the output video dir
* `<output_video_name>`: path to the output video name
* `<start_time>`: start time of the video(e.g., 30)
* `<end_time>`: end time of the video(e.g., 60)

