## main code 

from ScorePostProcessing import ScoreboardAnylazer
from VideoProcessing import VideoFrameIterator
from ScoreboardIdentifier import TextIdentifier


def main():
    
    video_processing = VideoFrameIterator("match01.mp4", "match01_paddle_output", "stream", "full")
    score_identifier = TextIdentifier()
    
    frame_count = 0
    while frame_count < 1100: 
        frame_count += 1
        with open("match01_output.txt", "a") as file:

            try:

                frame_path = next(video_processing)
                
                score_identifier.update_image(frame_path)

                ok, result = score_identifier.call_ocr() 

                if ok and all(result): 
                    for line in result:

                        file.write(frame_path + ":")
                        for text in line: 
                            file.write(text[1][0] + " ")  # Adding a newline character at the end
                        file.write("\n")

            except StopIteration:
                # StopIteration is raised when the iterator is exhausted
                print("Iterator is exhausted.")
                break
      
    anylasis_test = ScoreboardAnylazer("file", "match01_output.txt")
    anylasis_test.anyalze_scores()



if __name__ == "__main__":
    main()