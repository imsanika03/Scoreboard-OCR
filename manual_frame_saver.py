import cv2
import os

# Define a callback function to save the frame on mouse click
def save_frame(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, output_dir = param
        frame_name = os.path.join(output_dir, f"frame_{cv2.getTickCount()}.png")
        cv2.imwrite(frame_name, frame)
        print(f"Frame saved to {frame_name}")

def main(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Create a named window and set mouse callback
    cv2.namedWindow("Video Player")
    frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        # Display the video frame
        cv2.imshow("Video Player", frame)
        
        # Set the mouse callback with the current frame
        cv2.setMouseCallback("Video Player", save_frame, param=(frame, output_dir))
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video player.")
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "match02.mp4"  # Replace with your video file path
    output_dir = "match02_saved"  # Replace with your desired output directory
    main(video_path, output_dir)
