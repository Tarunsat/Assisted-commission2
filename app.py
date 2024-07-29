from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 segmentation model
segmentation_model = YOLO('best.pt')
# Load the YOLOv8 object detection model
detection_model = YOLO('best_obj2.pt')

# Function to get the center of a bounding box
def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/images/uploaded_image.jpg'
        file.save(file_path)
        return redirect(url_for('process_image'))
    return render_template('index.html')

@app.route('/process_image', methods=['GET', 'POST'])
def process_image():
    if request.method == 'POST':
        correct_values = request.form.to_dict()
        return render_template('display.html', values=correct_values)
    
    image = cv2.imread('static/images/uploaded_image.jpg')
    segmentation_results = segmentation_model(image)
    segmentation_bboxes = segmentation_results[0].boxes.xyxy.cpu().numpy()

    detection_results = detection_model(image)
        
                     
    for detection_result in detection_results:
        detection_bboxes = detection_results[0].boxes.xyxy.cpu().numpy()
        detection_confidences = detection_results[0].boxes.conf.cpu().numpy()
        detection_labels = detection_results[0].boxes.cls.cpu().numpy()

    red_boxes = []
        
    for segmentation_bbox in segmentation_bboxes:
        sx1, sy1, sx2, sy2 = segmentation_bbox
        for detection_bbox, confidence, label in zip(detection_bboxes, detection_confidences, detection_labels):
            center_x, center_y = get_bbox_center(detection_bbox)
            inside_segmentation = False
            if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                inside_segmentation = True
                break

        if inside_segmentation and confidence >= 0.50:
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, detection_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{detection_result.names[label]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = (0, 0, 255)
            red_boxes.append(detection_bbox)
            sx1, sy1, sx2, sy2 = map(int, segmentation_bbox)
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(image, f"Box number:{len(red_boxes-1)}", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite('static/images/marked_image.jpg', image)

    return render_template('correct_values.html', red_boxes=red_boxes, enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
