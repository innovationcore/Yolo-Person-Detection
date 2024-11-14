import cv2
from ultralytics import YOLO
import time

# Load the trained YOLO model
model = YOLO(r"Yolo-Person-Detection\CustomTrainedFallDetectModel\best.pt")  # Update with the path to your saved model

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define threshold for fall detection confidence
FALL_CONFIDENCE_THRESHOLD = 0.8  # Adjust based on model's performance during testing

print("Starting fall detection... Press 'q' to quit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Run YOLO model on the frame
        results = model(frame)

        # Process detections
        fallen_detected = False
        for detection in results[0].boxes:
            if int(detection.cls[0]) == 0:  # Assuming '0' is the class for 'person'
                confidence = detection.conf[0]
                bbox = detection.xywh[0]  # Get bounding box coordinates
                
                # Check if the model detects a fall with high confidence
                if confidence > FALL_CONFIDENCE_THRESHOLD:
                    fallen_detected = True
                    x, y, w, h = bbox.int().tolist()
                    
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fall Detected ({confidence:.2f})", (x - w // 2, y - h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # If person detected but not a fall, label as 'Person'
                    cv2.putText(frame, f"Person ({confidence:.2f})", (x - w // 2, y - h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add a message on the frame
        if fallen_detected:
            cv2.putText(frame, "ALERT: Fall Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('Fall Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
