import json
import numpy as np
import os
import argparse

# game_out_dir = './data/'
# with open(os.path.join(game_out_dir, 'pred-test.35.score_.json'), 'r') as infile:
#     data = json.load(infile)

import os
import json
import numpy as np
from typing import Dict, List

def score2npy(data: Dict[str, List[List[float]]], output_dir: str) -> None:
    """
    This function converts the provided JSON data into numpy files and stores them in the output directory 
    following the specified folder structure.
    
    Args:
        data (Dict[str, List[List[float]]]): A dictionary where each key is the game name
        and the value is a 2D list of model predictions for each frame.
        output_dir (str): The output directory where the numpy files will be saved.

    Returns:
        None. The function writes numpy files.
    """
    # Loop over each game
    for game, predictions in data.items():
        game_name, half = game.rsplit('/', 1)
        # Create directory structure
        game_dir = os.path.join(output_dir, game_name)
        os.makedirs(game_dir, exist_ok=True)

        # Write numpy array to file
        np.save(os.path.join(game_dir, f"{half}.npy"), np.array(predictions))
        

def main():
    parser = argparse.ArgumentParser(
        description="create numpy files from score json files"
    )

    parser.add_argument(
        "--pred_file",
        type=str,
        default="/outputs/pred-test.66.score.json",
        help="Path to the score json file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/predictions/test",
        help="Path to the output directory",
    )
    args = parser.parse_args()
    # Load data from JSON file
    data_path = args.pred_file
    with open(data_path, 'r') as infile:
        data = json.load(infile)

    # Convert data to numpy files
    score2npy(data, args.output_dir)
    
if __name__ == "__main__":
    main()
