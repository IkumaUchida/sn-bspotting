import copy
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns


class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()


class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                     self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom


def process_frame_predictions(
        dataset, classes, pred_dict, high_recall_score_threshold=0.01
):
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    for video, (scores, support) in sorted(pred_dict.items()):
        label = dataset.get_labels(video)
        # support[support == 0] = 1   # get rid of divide by zero
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})

    return err, f1, pred_events, pred_events_high_recall, pred_scores


def non_maximum_supression(pred, window):
    new_pred = []
    for video_pred in pred:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        for v in events_by_label.values():
            for e1 in v:
                for e2 in v:
                    if (
                            e1['frame'] != e2['frame']
                            and abs(e1['frame'] - e2['frame']) <= window
                            and e1['score'] < e2['score']
                    ):
                        # Found another prediction in the window that has a
                        # higher score
                        break
                else:
                    events.append(e1)
        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred


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
