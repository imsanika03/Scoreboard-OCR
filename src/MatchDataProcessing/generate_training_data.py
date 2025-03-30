
import os
import json

from MatchWinnerProcessing import MatchWinnerProcessor
from MatchTrainingData import MatchRawProcessor



def main(): 

    ## for a file in the post file directory
    directory = "/Users/sanikabharvirkar/Documents/pprlastshot/MatchDataProcessing/"

    post_directory = os.path.join(directory, "post")
    data_directory = os.path.join(directory, "data")
    training_data_directory = os.path.join(directory, "training_data")

    for filename in os.listdir(post_directory):
        if filename.endswith(".txt"):  # Process only `.txt` files
            post_file = os.path.join(post_directory, filename)
            match_winner = MatchWinnerProcessor(post_file)

            for match_name, winner in match_winner.process_file():
                    filename = f"{match_name}.npy"
                    data_file_path = os.path.join(data_directory, filename)

                    match_data = MatchRawProcessor(data_file_path)
                    player1_trajectory, player2_trajectory, player1_actions, player2_actions = match_data.build_states_and_actions()
                    print(type(player1_trajectory))
                    # generalized match_winner to put in file
                    training_data = None
                    if winner == match_winner.player1_name:
                        training_data = [
                            {
                                "winner": "P1",
                                "winner_trajectories": player1_trajectory,
                                "winner_actions": player1_actions,
                                "loser": "P2",
                                "loser_trajectories": player2_trajectory,
                                "loser_actions": player2_actions
                            }
                        ]
                    else: 
                        training_data = [
                            {
                                "winner": "P2",
                                "winner_trajectories": player2_trajectory,
                                "winner_actions": player2_actions,
                                "loser": "P1",
                                "loser_trajectories": player1_trajectory,
                                "loser_actions": player1_actions
                            }
                        ]
                         

                    # Save to JSON file
                    training_filename = f"{match_name}.json"
                    training_file_path = os.path.join(training_data_directory, training_filename)

                    with open(training_file_path, 'w') as f:
                        json.dump(training_data, f, indent=4)

if __name__ == "__main__":
    main()