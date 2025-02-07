## main code 

from video_processor import VideoFrameIterator
from scoreboard_identifier import TextIdentifier


def main():
    
    video_processing = VideoFrameIterator("match02.mp4", "match02_paddle_output", "save", "full")
    score_identifier = TextIdentifier()
    
    count = 0 
    while count < 20:
        with open("output_video2.txt", "a") as file:
    
            try:

                frame_path = next(video_processing)
                
                score_identifier.update_image(frame_path)

                ok, result = score_identifier.call_ocr() 

                if ok and all(result): 
                    for line in result:
                        if len(line) == 6: 
                            file.write(frame_path + ":")
                            for text in line: 
                                file.write(text[1][0] + " ")  # Adding a newline character at the end
                            file.write("\n")


            except StopIteration:
                # StopIteration is raised when the iterator is exhausted
                print("Iterator is exhausted.")
                break




if __name__ == "__main__":
    main()