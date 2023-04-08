import argparse
import cv2
import json
import numpy as np
from pathlib import Path

import numpy as np

def draw_bar_chart(frame, drive_confidence, pass_confidence, max_confidence=1.0):
    bar_chart_width = 750
    chart_height = 60
    chart_x = 50
    chart_y = frame.shape[0] - chart_height * 2 - 20

    # Draw the background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (chart_x, chart_y), (chart_x + bar_chart_width, chart_y + chart_height * 2), (0, 0, 0), -1)

    # Draw the bars
    drive_bar_width = int((drive_confidence / max_confidence) * bar_chart_width)
    pass_bar_width = int((pass_confidence / max_confidence) * bar_chart_width)

    cv2.rectangle(overlay, (chart_x, chart_y), (chart_x + drive_bar_width, chart_y + chart_height), (0, 0, 255), -1)
    cv2.rectangle(overlay, (chart_x, chart_y + chart_height), (chart_x + pass_bar_width, chart_y + chart_height * 2), (255, 191, 0), -1)

    # Add transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the labels
    cv2.putText(frame, "Drive", (chart_x - 20, chart_y + chart_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Pass", (chart_x - 20, chart_y + chart_height * 3 // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw scale lines and values
    scale_interval = 0.2
    num_intervals = int(max_confidence / scale_interval)
    for i in range(1, num_intervals + 1):
        x_pos = chart_x + int(i * scale_interval * bar_chart_width)
        cv2.line(frame, (x_pos, chart_y - 5), (x_pos, chart_y + chart_height * 2 + 5), (255, 255, 255), 1)
        cv2.putText(frame, str(np.round(i * scale_interval, 1)), (x_pos - 10, chart_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def put_text_with_background(frame, text, position, font, font_scale, color, thickness):
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x-10, y - size[1]-10), (x + size[0]+10, y+10), (0, 0, 0), -1)
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


def main(input_video, input_json, output_video_dir, output_video_name, start, end):
    # Create output directory if it doesn't exist
    output_dir = Path(output_video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the JSON file
    with open(input_json) as f:
        data = json.load(f)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    # Convert position to frame number and store it as keys with label and confidence as values
    events = {}
    for entry in data["predictions"]:
        frame_number = int(int(entry["position"]) / 1000 * fps)
        events[frame_number] = (entry["label"], float(entry["confidence"]))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize the VideoWriter object
    output_video_path = output_dir / output_video_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change the codec to 'mp4v'
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_number = start_frame
    label_display_frames_left = 0
    label = None
    drive_confidence = 0
    pass_confidence = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_number > end_frame:
            break

        if frame_number in events:
            label, confidence = events[frame_number]
            label_display_frames_left = fps // 2  # Display label for half a second

            if label == "DRIVE":
                drive_confidence = confidence
                pass_confidence = 0
            else:
                drive_confidence = 0
                pass_confidence = confidence

        if label_display_frames_left > 0:
            put_text_with_background(frame, label, (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            label_display_frames_left -= 1

        draw_bar_chart(frame, drive_confidence, pass_confidence)

        out.write(frame)
        print(f"Writing frame {frame_number}")

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video successfully created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_video_dir", type=str, required=True, help="Directory to save the output video.")
    parser.add_argument("--output_video_name", type=str, required=True, help="Name of the output video file.")
    parser.add_argument("--start", type=int, default=0, help="Start time (in seconds) of the output video.")
    parser.add_argument("--end", type=int, default=60, help="End time (in seconds) of the output video.")
    args = parser.parse_args()

    main(args.input_video, args.input_json, args.output_video_dir, args.output_video_name, args.start, args.end)
