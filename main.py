
import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 model 
# (It will automatically download 'yolov8n.pt' the first time you run this)
# 'n' stands for nano - it's the fastest model, perfect for video!
model = YOLO('yolov8n.pt') 

# 2. Open the video file 
# Note: Ensure you have a video named 'sample_input.mp4' in the same folder.
# Tip: Change this to 0 (i.e., cv2.VideoCapture(0)) to use your webcam!
video_path = 'sample_input.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file. Check the file name and path.")
    exit()

# 3. Get video properties to save the output correctly
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the VideoWriter to save the result
# 'mp4v' is the codec used to write .mp4 files
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing video... A window will pop up. Press 'q' to stop early.")

# 4. The Processing Loop
while cap.isOpened():
    # Read a single frame
    success, frame = cap.read()
    
    if success:
        # Run YOLO object detection on that frame
        # stream=True keeps it fast and memory efficient
        results = model(frame, stream=True) 
        
        for result in results:
            # .plot() automatically draws the bounding boxes and labels for us!
            annotated_frame = result.plot()
        
        # Write the annotated frame to our output video file
        out.write(annotated_frame)
        
        # Display the frame on your screen so you can watch it happen
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # If success is False, the video has ended
        break

# 5. Clean up! Release the camera/video and close windows.
cap.release()
out.release()
cv2.destroyAllWindows()

print("Done! Check your folder for 'output_video.mp4'")
