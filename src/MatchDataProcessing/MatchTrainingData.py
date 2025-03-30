import numpy as np
import pandas as pd

LEFT_WRIST_IDX = 4
RIGHT_WRIST_IDX = 7

class MatchRawProcessor(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.player_1_name = None
        self.player_2_name = None
        self.player_1_wrist = None
        self.player_2_wrist = None
        self.winner = None 

        self.player1_hits = None
        self.player2_hits = None
        self.bounces = None
        self.fps = 10 
        
        self.load_data()
        self.player1_hits, self.player2_hits, self.bounces = self.get_player_hits()
        self.scene = self.get_scene_data()
        self.assign_paddle_hand()

        ## after this, the only variables left unintialized are wrists, and winner
    def load_data(self):
        self.data = np.load(self.file_path)

    def get_player_hits(self):
        player1_hits = self.data[:, 1, 0]  # First row in 'temp'
        player2_hits = self.data[:, 1, 1]  # Second row in 'temp'
        bounces = self.data[:, 1, 2]       # Third row in 'temp'
        return player1_hits, player2_hits, bounces
    

    def get_scene_data(self, min_frame=0, max_frame=58):
        return {
            i: (self.data[i, 2:46, :], self.data[i, 46:90, :], self.data[i, 90:, :].flatten()) for i in range(min_frame, max_frame)
        }
    
    def assign_paddle_hand(self): 

        ## Player 1 
        
        idxPlayer1_hit = np.where(self.player1_hits == 1)[0]
        p1_keypoints, _, ball_pos = self.scene[idxPlayer1_hit[0]]

        ball = np.array(ball_pos)
        if np.linalg.norm(ball - p1_keypoints[LEFT_WRIST_IDX]) <  np.linalg.norm(ball - p1_keypoints[RIGHT_WRIST_IDX]): 
            self.player_1_wrist = LEFT_WRIST_IDX
        else: 
            self.player_1_wrist = RIGHT_WRIST_IDX

        ## Player 2 

        idxPlayer2_hit = np.where(self.player2_hits == 1)[0]
        _, p2_keypoints, ball_pos = self.scene[idxPlayer2_hit[0]]

        ball = np.array(ball_pos)
        if np.linalg.norm(ball - p2_keypoints[LEFT_WRIST_IDX]) <  np.linalg.norm(ball - p2_keypoints[RIGHT_WRIST_IDX]): 
            self.player_2_wrist = LEFT_WRIST_IDX
        else: 
            self.player_2_wrist = RIGHT_WRIST_IDX



    def get_next_ball_positions(self, scene, frame, num_positions=3):
        next_positions = [
            scene[i][2] for i in range(frame + 1, frame + 1 + num_positions) if i in scene
        ]
        return next_positions

    def build_states_and_actions(self): 

        print("Building player 1 trajectory")

        scene = self.scene
        ## for first 1 in player1_hits, player2_hits, initialize the start of the sequence. 

        ## ex if the first is in player1_hits, the third bounce is part of player1 trajectory (action), so is the fifth , etc 

        player1_index = next((i for i, x in enumerate(self.player1_hits) if x != 0), None)
        player2_index = next((i for i, x in enumerate(self.player2_hits) if x != 0), None)
        bounce_index = next((i for i, x in enumerate(self.bounces) if x != 0), None)

        if bounce_index > player1_index and bounce_index > player2_index:
            ## move to start of min player bounce 
            self.bounces = self.bounces[min(player1_index, player2_index):]

        player1_trajectory = []
        player2_trajectory = []

        player1_actions = [] 
        player2_actions = []
        last_bounce = False
        
        ## state: [ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, paddle_x, paddle_y, paddle_z]
        ## action: [target_x, target_y]
        
        next_bounce_player1 = False
        if player1_index < player2_index: 
            next_bounce_player1 = True

        for frame in scene:
            p1_keypoints, p2_keypoints, ball_pos = scene[frame]
            ## player 1 starts the rally 
            if self.player1_hits[frame] == 1:
                ball_position = ball_pos

                # Time step per frame
                next_ball_positions = self.get_next_ball_positions(scene, frame)

                dt = len(next_ball_positions) / self.fps
                v_x = (next_ball_positions[-1][0] - ball_pos[0]) / dt
                v_y = (next_ball_positions[-1][1] - ball_pos[1]) / dt
                v_z = (next_ball_positions[-1][2] - ball_pos[2]) / dt

                paddle_position = p1_keypoints[self.player_1_wrist]
                print("appending player 1 trajectory")
                player1_trajectory.append(([ball_position[0], ball_position[1], ball_position[2]]
                                           ,[v_x, v_y, v_z], 
                                           [paddle_position[0], paddle_position[1], paddle_position[2]]))
                next_bounce_player1 = True 
                last_bounce = False
            elif self.player2_hits[frame] == 1:

                ball_position = ball_pos

                # Time step per frame
                next_ball_positions = self.get_next_ball_positions(scene, frame)

                dt = len(next_ball_positions) / self.fps
                v_x = (next_ball_positions[-1][0] - ball_pos[0]) / dt
                v_y = (next_ball_positions[-1][1] - ball_pos[1]) / dt
                v_z = (next_ball_positions[-1][2] - ball_pos[2]) / dt

                paddle_position = p2_keypoints[self.player_2_wrist]
                player2_trajectory.append(([ball_position[0], ball_position[1], ball_position[2]]
                            ,[v_x, v_y, v_z], 
                            [paddle_position[0], paddle_position[1], paddle_position[2]]))
                next_bounce_player1 = False
                last_bounce = False

            elif self.bounces[frame] == 1:
                ## this is to prevent double bounces from being added like in a serve
                if last_bounce:
                    continue
                if next_bounce_player1:
                    player1_actions.append([ball_pos[0], ball_pos[1], ball_pos[2]])
                    next_bounce_player1 = False
                    last_bounce = True
                else: 
                    player2_actions.append([ball_pos[0], ball_pos[1], ball_pos[2]])
                    last_bounce = True

        return player1_trajectory, player2_trajectory, player1_actions, player2_actions

def main():
    # Example data (scene is a dictionary with frame numbers as keys)
    file_path = "/Users/sanikabharvirkar/Documents/pprlastshot/match1_5.npy"
    processor = MatchRawProcessor(file_path)
    player1_trajectory, player2_trajectory, player1_actions, player2_actions = processor.build_states_and_actions()

    print(processor.player1_hits)
    print(processor.player2_hits)
    print(processor.bounces)

    ## verify accuracy 

    for frame in processor.scene: 
        p1_keypoints, p2_keypoints, ball_pos = processor.scene[frame]
        if processor.player1_hits[frame] == 1:
            print("Player 1 hit")
            print(ball_pos)
        if processor.player2_hits[frame] == 1:
            print("Player 2 hit")
            print(ball_pos)
        if processor.bounces[frame] == 1:
            print("Bounce")
            print(ball_pos)

if __name__ == "__main__":
    main()

#         return player1_trajectory, player2_trajectory, player1_actions, player2_actions



#         # for frame in scene: 
#         #     p1_keypoints, p2_keypoints, ball_pos = scene[frame]
#         #     if player1_hits[frame] == 1:
#         #         print("Player 1 hit")
#         #         print(ball_pos)
#         #     if player2_hits[frame] == 1:
#         #         print("Player 2 hit")
#         #         print(ball_pos)
#         #     if bounces[frame] == 1:
#         #         print("Bounce")
#         #         print(ball_pos)
    
    

# # Example of storing trajectories with additional labels like player or hit type
# columns = ['timestamp', 'ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz', 'target_x', 'target_y',
#            'paddle_x', 'paddle_y', 'paddle_z', 'hit_type', 'player_id']
# data = []

# output = np.load("/Users/sanikabharvirkar/Documents/pprlastshot/match1_5.npy")

# player1_hits = output[:, 1, 0]  # First row in 'temp'
# player2_hits = output[:, 1, 1]  # Second row in 'temp'
# bounces = output[:, 1, 2]       # Third row in 'temp'


# min_frame, max_frame = 0, 58
# scene = {
#     i: (output[i, 2:46, :], output[i, 46:90, :], output[i, 90:, :].flatten()) for i in range(max_frame)
#     }


# for frame in scene: 
#     p1_keypoints, p2_keypoints, ball_pos = scene[frame]
#     if player1_hits[frame] == 1:
#         print("Player 1 hit")
#         print(ball_pos)
#     if player2_hits[frame] == 1:
#         print("Player 2 hit")
#         print(ball_pos)
#     if bounces[frame] == 1:
#         print("Bounce")
#         print(ball_pos)

# # Convert to DataFrame
# df = pd.DataFrame(data, columns=columns)

# # To parse the trajectory for a specific player
# player_0_trajectory_0 = df[(df['player_id'] == 0) & (df['timestamp'] < 5)]

# # Extract states for a given player
# states_for_player_0 = df[df['player_id'] == 0][['ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz', 
#                                                 'paddle_x', 'paddle_y', 'paddle_z']].values
