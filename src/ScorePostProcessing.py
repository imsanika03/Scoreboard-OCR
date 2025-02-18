import os
import re

class ScoreboardAnylazer: 

    def __init__(self, file_type, input_path, output_path=None):
        
        if file_type == "file": # single output file
            self.file_type = "file"

        elif file_type == "dir": 
            self.file_type = "dir"
        else: 
            raise ValueError("File Type must be either 'file' or 'dir'")
        
        self.input_path = input_path

        if output_path: 

            self.output_path = output_path
        else: 
            print("no output path")
            self.output_path = self.input_path.replace(".txt", "") + "_post.txt"

    def anyalze_scores(self): 
        
        self.previous_scores = {}

        if self.file_type == "dir": 
            self.anyalze_scores_dir()
        
        elif self.file_type == "file":
            self.anyalze_scores_file(self.input_path)

        else: 
            raise NotImplementedError("Not implemented yet...")

    def anyalze_scores_dir(self): 
        
        files = [f for f in os.listdir(self.input_path) if "_" in f and f.endswith(".txt")]

         # Sort files numerically based on the number after the underscore
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        for filename in files:
    
            filepath = os.path.join(self.input_path, filename)
            if os.path.isfile(filepath):  # Ensure it's a file, not a directory
                print("reading", filepath, "is not a file path")
                self.anyalze_scores_file(filepath)
                print(self.previous_scores)
            else: 
                print(filepath, "is not a file path")
    
    def anyalze_scores_file(self, filename): 
        
        output_file = self.output_path
        previous_scores = self.previous_scores

        with open(filename, "r") as file, open(output_file, "a") as outfile:
            
            for line in file:

                parts = line.strip().split(":")

                if len(parts) != 2:
                    print("malformed line")
                    continue  # Skip malformed lines

                frame_info, score_data = parts
                frame_number = re.search(r"frame_(\d+)", frame_info)
                if not frame_number:
                    print("frame num not found")
                    continue  # Skip if frame number isn't found
                frame = frame_number.group(1)

                score_tokens = score_data.split()
                print(len(score_tokens))

                score_tokens = self.clean_tokens(score_tokens)

                if score_tokens is False: 
                    continue # error with formatting
                
                score_tokens = [score_tokens[0], int(score_tokens[1]), int(score_tokens[2]), 
                                score_tokens[3], int(score_tokens[4]), int(score_tokens[5])]
                
                team1, score1, score2, team2, score3, score4 = score_tokens

                new_scores = {team1: (score1, score2), team2: (score3, score4)}

                # Compare with previous scores and log changes
                for team, new_score in new_scores.items():
                    if team in previous_scores and previous_scores[team] != new_score:
                        if abs(sum(new_scores[team]) - sum(previous_scores[team])) == 1: # score changes by max 1
                            
                            outfile.write(f"{filename}, {frame}, {new_scores}\n")
                
                # Update previous scores
                previous_scores = new_scores
        
        self.previous_scores = previous_scores

    ''' checks if tokens follows formal <name> <score1> <score2> <name> <score1> <score2>'''
    def validate_score_data(self, score_data):
        
        if len(score_data) != 6:
            return False

        def is_number(s):
            try:
                float(s)  
                return True
            except ValueError:
                return False

        return (
            not is_number(score_data[0]) and  
            is_number(score_data[1]) and      
            is_number(score_data[2]) and      
            not is_number(score_data[3]) and  
            is_number(score_data[4]) and      
            is_number(score_data[5])          
        )

    def clean_tokens(self, score_tokens): 
        
        # no post-processing required
        if len(score_tokens) == 6 and self.validate_score_data(score_tokens): 
            return score_tokens
        elif len(score_tokens) == 6: # case where validation fails because score not a float
            
            team1, score1, score2, team2, score3, score4 = score_tokens

            score_tokens = [s.replace("O", "0") for s in [score1, score2, score3, score4]]

            if not self.validate_score_data(score_tokens): ## bigger issue with score format if failing here
                return False
            else: 
                return score_tokens

        elif len(score_tokens) == 8: 
            score_tokens[0] = score_tokens[0] + "/" + score_tokens[1]
            score_tokens[4] = score_tokens[4] + "/" + score_tokens[5]
            score_tokens = [score_tokens[0]] + score_tokens[2:4] + [score_tokens[4]] + score_tokens[6:]
            
            #recursive call on score_tokens after conctenating score tokens to size 6
            # use case: TOMOKAZU HARIMOTO 0 1 LIN SHIDONG 0 0 instead of TOMOKAZU/HARIMOTO 0 1 LIN/SHIDONG 0 0 
            return self.clean_tokens(score_tokens)
        else: 
            return False

def main(): 

    anylasis_test = ScoreboardAnylazer("file", "output.txt")
    anylasis_test.anyalze_scores()

    #diretory usage 
    anylasis_test = ScoreboardAnylazer("dir", "match01_outputs")
    anylasis_test.anyalze_scores()

if __name__ == "__main__":
    main()