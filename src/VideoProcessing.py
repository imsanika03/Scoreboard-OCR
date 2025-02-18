import cv2
import os

class VideoFrameIterator:

    # mode: stream deletes every frame
    #       save saves every 10th frame 
    #
    # image_size: quadrant saves lower left quadrant of frame 

    def __init__(self, video_path, output_folder, mode="stream", image_size="quadrant"):
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_count = 0
        self.cap = cv2.VideoCapture(video_path)

        self.current_frame_path = None

        self.mode = mode
        self.image_size = image_size

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video file.")
        
    def __iter__(self):
        return self

    def __next__(self):

        if self.mode == "stream": 

            #delete previous frame 
            self.delete_last_frame()

        elif self.mode == "save": 

            if self.frame_count % 10 != 0: 
                self.delete_last_frame()
       
        # Read the next frame
        ret, frame = self.cap.read()
        
        if not ret:  # End of video
            self.cap.release()
            raise StopIteration
        
        frame_filename = os.path.join(self.output_folder, f"frame_{self.frame_count:04d}.jpg")
       
        if self.image_size == "quadrant": 
        
            # Get frame dimensions and extract the lower-right quadrant
            height, width, _ = frame.shape
            lower_left_quadrant = frame[height // 2:, :width // 2]

            # Save the frame temporarily
            
            cv2.imwrite(frame_filename, lower_left_quadrant)
            print(f"Yielding frame: {frame_filename}")
        
        else: 
            cv2.imwrite(frame_filename, frame)
            print(f"Yielding frame: {frame_filename}")

        # Store the frame file path for deletion on next call
        self.last_frame_path = frame_filename
        self.frame_count += 1

        return frame_filename
        

    def delete_last_frame(self):
        """Deletes the last saved frame."""
        if hasattr(self, "last_frame_path") and os.path.exists(self.last_frame_path):
            os.remove(self.last_frame_path)
            print(f"Deleted frame: {self.last_frame_path}")
            del self.last_frame_path

# Usage example
if __name__ == "__main__":
    video_file = "example_video.mp4"  # Path to the video
    output_directory = "frames_temp"  # Temporary folder to store frames

    frame_iterator = VideoFrameIterator(video_file, output_directory)

    for frame in frame_iterator:
        # Process the frame here
        cv2.imshow("Frame", frame)  # Example: Display frame
        
        # Wait for a key press or quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Delete the previous frame after it's processed
        frame_iterator.delete_last_frame()

    # Cleanup
    cv2.destroyAllWindows()


