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

def update(image_path,correct_values):
    no_red_box=0
    red_boxes = {}
    image = cv2.imread(image_path)
    segmentation_results = segmentation_model(image)
    segmentation_bboxes = segmentation_results[0].boxes.xyxy.cpu().numpy()

    detection_results = detection_model(image)
        
                     
    for detection_result in detection_results:
        detection_bboxes = detection_results[0].boxes.xyxy.cpu().numpy()
        detection_confidences = detection_results[0].boxes.conf.cpu().numpy()
        detection_labels = detection_results[0].boxes.cls.cpu().numpy()

    # # Convert to integer labels for easier processing
    # detection_labels = [int(label) for label in detection_labels]

    # # Draw and update bounding boxes
    # for idx, detection_bbox in enumerate(detection_bboxes):
    #     confidence = detection_confidences[idx]
    #     label = detection_labels[idx]
    #     user_label = correct_values.get(f"box{idx}", str(label))
        
    #     color = (0, 255, 0)
    #     x1, y1, x2, y2 = map(int, detection_bbox)
    #     cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(image, f"{user_label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
    for segmentation_bbox in segmentation_bboxes:
        sx1, sy1, sx2, sy2 = segmentation_bbox
        for detection_bbox, confidence, label in zip(detection_bboxes, detection_confidences, detection_labels):
            center_x, center_y = get_bbox_center(detection_bbox)
            inside_segmentation = False
            if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                inside_segmentation = True
                break

        if inside_segmentation and confidence >= 0.40:
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, detection_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{detection_result.names[label]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = (0, 0, 255)
            red_boxes[no_red_box]=segmentation_bbox
            no_red_box= no_red_box+1
            # sx1, sy1, sx2, sy2 = map(int, segmentation_bbox)
            # cv2.rectangle(image, (sx1, sy1), (sx2, sy2), color, 2)
            # cv2.putText(image, f"Box {len(red_boxes)-1}", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    print(red_boxes)
    print(correct_values)
    for i in range(0,len(red_boxes)):
        color = (0, 255, 0)
        x1, y1, x2, y2 = map(int, red_boxes.get(i))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{correct_values[str(i)]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imwrite('static/images/marked_image.jpg', image)
    
    

def analyse(image_path):
    no_red_box=0
    red_boxes = {}
    
    image = cv2.imread(image_path)
    segmentation_results = segmentation_model(image)
    segmentation_bboxes = segmentation_results[0].boxes.xyxy.cpu().numpy()

    detection_results = detection_model(image)
        
                     
    for detection_result in detection_results:
        detection_bboxes = detection_results[0].boxes.xyxy.cpu().numpy()
        detection_confidences = detection_results[0].boxes.conf.cpu().numpy()
        detection_labels = detection_results[0].boxes.cls.cpu().numpy()

    
        
    for segmentation_bbox in segmentation_bboxes:
        sx1, sy1, sx2, sy2 = segmentation_bbox
        for detection_bbox, confidence, label in zip(detection_bboxes, detection_confidences, detection_labels):
            center_x, center_y = get_bbox_center(detection_bbox)
            inside_segmentation = False
            if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                inside_segmentation = True
                break

        if inside_segmentation and confidence >= 0.40:
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, detection_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{detection_result.names[label]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = (0, 0, 255)
            red_boxes[no_red_box]=segmentation_bbox
            sx1, sy1, sx2, sy2 = map(int, segmentation_bbox)
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(image, f"Box {no_red_box}", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            no_red_box= no_red_box+1

    cv2.imwrite('static/images/marked_image.jpg', image)
    # print(red_boxes)
    return red_boxes

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
        # print(correct_values)

        update('static/images/uploaded_image.jpg',correct_values)
        return render_template('display.html', values=correct_values)
        
    return render_template('correct_values.html', red_boxes=analyse('static/images/uploaded_image.jpg'), enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
