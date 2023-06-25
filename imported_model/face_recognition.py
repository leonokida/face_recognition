import cv2
import numpy as np

# Load YOLO v3 from Darknet
net = cv2.dnn.readNetFromDarknet("yolov3/yolov3.cfg", "yolov3/yolov3.weights")

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = []
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Open webcam
cap = cv2.VideoCapture(0)

# without NMS
"""
while True:
    ret, frame = cap.read()  # Read frame from the webcam

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the frame through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 represents faces
                # Extract bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Draw bounding box and label on the frame
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
"""

# With NMS
while True:
    ret, frame = cap.read()

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the frame through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the output
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 represents faces
                # Extract bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate top-left corner coordinates of the face region
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Adjust bounding box dimensions and position for face region
                face_x = max(0, x)
                face_y = max(0, y)
                face_width = min(frame.shape[1] - 1, x + width) - face_x
                face_height = min(frame.shape[0] - 1, y + height) - face_y

                # Store detection information
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([face_x, face_y, face_width, face_height])


    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw the bounding boxes and labels
    for i in range(len(boxes)):
        if i in indices:
            x, y, width, height = boxes[i]
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

# Release the resources
cap.release()
cv2.destroyAllWindows()