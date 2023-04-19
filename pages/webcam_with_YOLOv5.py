import streamlit as st
import cv2
import numpy as np

# Load the YOLOv5 model
model = cv2.dnn.readNet("../yolov5s.pt", "../setup.cfg")

# Set the threshold for detection confidence
conf_threshold = 0.5

# Define the function to capture video from the webcam
def capture_video():
    # Create a video capture object for the webcam
    cap = cv2.VideoCapture(0)

    # Check if the capture object is successfully opened
    if not cap.isOpened():
        st.error("Unable to open camera.")
        return

    # Loop to process frames from the captured video
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # If failed to read a frame, break the loop
        if not ret:
            break

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set the input to the model
        model.setInput(blob)

        # Perform object detection
        detections = model.forward()

        # Loop through the detections
        for i in range(detections.shape[2]):
            # Get the detection confidence
            confidence = detections[0, 0, i, 4]

            # If the confidence is greater than the threshold
            if confidence > conf_threshold:
                # Get the object label and bounding box coordinates
                class_id = int(detections[0, 0, i, 1])
                label = str(class_id)
                left = int(detections[0, 0, i, 0] * frame.shape[1])
                top = int(detections[0, 0, i, 1] * frame.shape[0])
                right = int(detections[0, 0, i, 2] * frame.shape[1])
                bottom = int(detections[0, 0, i, 3] * frame.shape[0])

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result image
        cv2.imshow("Object Detection", frame)

        # Wait for a key event
        key = cv2.waitKey(1)

        # If 'q' key is pressed, break the loop
        if key == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

# Define the main function to create a Streamlit app
def main():
    st.title("Real-time Object Detection with YOLOv5 and Webcam")

    # Create a button to start capturing video
    if st.button("Start Webcam"):
        # Call the function to capture video
        capture_video()

if __name__ == "__main__":
    main()
