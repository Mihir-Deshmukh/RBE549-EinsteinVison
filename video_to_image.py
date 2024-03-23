import cv2
import os

def read_frames_from_video(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0

    # Read frames from the video until it is complete
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame was not successfully read, the video is complete
        if not ret:
            break

        # Save the frame as an image
        if frame_count % 10 == 0:
            # Save the frame as an image
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count}.png'), frame)

        frame_count += 1

    # Release the video file
    video.release()

# Path to the video file
# video_path = "P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4"
# video_path = "P3Data/Sequences/scene11/Undist/2023-03-11_17-19-53-front_undistort.mp4"
video_path = "P3Data/Sequences/scene9/Undist/2023-03-04_17-20-36-front_undistort.mp4"

output_dir = "output_frames/scene_9"

# Call the function to read frames from the video
read_frames_from_video(video_path, output_dir)