import cv2

import easyocr
from ultralytics import YOLO

ocr = easyocr.Reader(['en'], gpu=True)
# Load the YOLOv8 model
model = YOLO(r"C:\Users\piyus\PycharmProjects\pythonProject7\runs\detect\train29\weights\best.pt")

# Open the video file
video_path = r"C:\Users\piyus\Downloads\170609_A_Delhi_026.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)


        # Visualize the results on the frame
        annotated_frame = results[0].plot()



        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()