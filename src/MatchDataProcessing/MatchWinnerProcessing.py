import re
import ast


class MatchWinnerProcessor:
    def __init__(self, post_file):
        self.post_file = post_file
        self.refreshed = False
        self.referesh_test = []
        self.pattern = r"/([^/]+)\.txt,\s*\d+,\s*(\{.*\})"  
        self.player1_name = None 
        self.player2_name = None

    def process_file(self):
        """ Generator that streams match results as they are processed. """
        with open(self.post_file, "r") as f:
            for line in f:
                match = re.search(self.pattern, line)
                if match:
                    yield from self.process_line(match)

    def process_line(self, match):
        """ Process a single line and yield match results. """
        match_name = match.group(1).replace(".txt", "")  
        scores_dict = ast.literal_eval(match.group(2))  

        # Extract players and scores
        players = list(scores_dict.keys())
        scores = list(scores_dict.values())

        player1, player2 = players

        if (self.player1_name is None) and (self.player2_name is None):
            self.player1_name = player1
            self.player2_name = player2
            
        player1_match_point, player1_rally_point = map(int, scores[0])
        player2_match_point, player2_rally_point = map(int, scores[1])

        if not self.refreshed:
            self.referesh_test = [player1, player1_match_point, player1_rally_point, 
                                  player2, player2_match_point, player2_rally_point, match_name]
            self.refreshed = True
            print("Refreshed:", self.referesh_test)
        else:
            yield from self.update_scores(player1, player1_match_point, player1_rally_point, 
                                          player2, player2_match_point, player2_rally_point, match_name)

    def update_scores(self, player1, player1_match_point, player1_rally_point, 
                      player2, player2_match_point, player2_rally_point, match_name):
        """ Determines who won the rally and yields the result. """
        if self.referesh_test[1] == player1_match_point and self.referesh_test[4] == player2_match_point:
            if self.referesh_test[2] == player1_rally_point and self.referesh_test[5] < player2_rally_point:
                yield match_name, player2  # Player 2 wins the rally
                self.referesh_test[6] = match_name
                self.referesh_test[5] = player2_rally_point

            elif self.referesh_test[2] < player1_rally_point and self.referesh_test[5] == player2_rally_point:
                yield match_name, player1  # Player 1 wins the rally
                self.referesh_test[6] = match_name
                self.referesh_test[2] = player1_rally_point

            else:  # Invalid match
                self.reset_refresh()
        else:
            self.reset_refresh()  # Match point changed, refresh state

    def reset_refresh(self):
        """ Resets tracking variables when match points change. """
        self.refreshed = False
        self.referesh_test = []

# Example Usage
# post_file = "/Users/sanikabharvirkar/Documents/pprlastshot/match1_post.txt"
# processor = MatchWinnerProcessor(post_file)

# Streaming results
# for match_name, winner in processor.process_file():
#     print(f"Match: {match_name}, Winner: {winner}")




