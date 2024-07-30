from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
threshold =0.1

detection_big = YOLO('best_bigobj.pt')
# Load the YOLOv8 segmentation model
segmentation_model = YOLO('best.pt')
# Load the YOLOv8 object detection model
detection_model = YOLO('best_obj6.pt')

# Function to get the center of a bounding box
def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def update(image_path,correct_values):
    no_red_box=0
    index =0
    red_boxes = {}
    all_boxes={}
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

        if inside_segmentation and confidence >= threshold:
            color = (255, 0, 0)
            x1, y1, x2, y2 = map(int, detection_bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{detection_result.names[label]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            all_boxes[index] = [detection_result.names[label], detection_bbox]
            index=index+1
        else:
            color = (0, 0, 255)
            red_boxes[no_red_box]=segmentation_bbox
            no_red_box= no_red_box+1
            
            # sx1, sy1, sx2, sy2 = map(int, segmentation_bbox)
            # cv2.rectangle(image, (sx1, sy1), (sx2, sy2), color, 2)
            # cv2.putText(image, f"Box {len(red_boxes)-1}", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # print(red_boxes)
    # print(correct_values)
    for i in range(0,len(red_boxes)):
        color = (255, 255, 0)
        x1, y1, x2, y2 = map(int, red_boxes.get(i))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{correct_values[str(i)]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        all_boxes[index] = [correct_values[str(i)], red_boxes.get(i)]
        index=index+1
        
    cv2.imwrite('static/images/marked_image.jpg', image)
    ordering(all_boxes,'static/images/cropped.jpg')
    # print(all_boxes)
    
    
    
def crop(image_path):
    image = cv2.imread(image_path)
    detection_results = detection_big(image)
        
                     
    for detection_result in detection_results:
        detection_bboxes = detection_results[0].boxes.xyxy.cpu().numpy()
        # detection_bboxes = detection_results[0].boxes.xyxy  # Extract bounding boxes
        x1, y1, x2, y2 = map(int, detection_bboxes[0])  # Convert bounding box to integers

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = 0
        x2 = min(image.shape[1], x2)
        y2 = image.shape[0]

        cropped_image = image[y1:y2, x1:x2]  # Crop the image to the bounding box
    cv2.imwrite('static/images/cropped.jpg', cropped_image)
        
    
    
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
        inside_segmentation = False
        for detection_bbox, confidence, label in zip(detection_bboxes, detection_confidences, detection_labels):
            center_x, center_y = get_bbox_center(detection_bbox)
            
            if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                inside_segmentation = True
                break

        if inside_segmentation and confidence >= threshold:
            color = (255, 0, 0)
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

def ordering(all_detections,image_path):
    TopHalf={}
    BottomHalf={}
    Xsorted = sorted(all_detections.items(), key=lambda item: item[1][1][0])
    Xsorted_dict = dict(Xsorted)
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    for k,v in Xsorted_dict.items():
        centerx,centery= get_bbox_center(v[1])
        if(centery<height/2):
            TopHalf[k] = v
        else:
            BottomHalf[k]=v
    

    
        
    print(TopHalf)
    print()
    print(BottomHalf)
    
    checking(TopHalf,BottomHalf,image_path)
    

def checking(TopHalf,BottomHalf,image_path):
    CorrectTopHalf=['1','3','5']
    image = cv2.imread(image_path)
    CorrectBottomHalf=['2','4','6']
    i=0
    for k,v in TopHalf.items():
        try:
            if(v[0]==CorrectTopHalf[i]):
                color = (0, 255, 0)
                x1, y1, x2, y2 = map(int, v[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{CorrectTopHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                color = (0, 0, 255)
                x1, y1, x2, y2 = map(int, v[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Supposed to be:{CorrectTopHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            i=i+1
        except:
            color = (0, 0, 255)
            x1, y1, x2, y2 = map(int, v[1])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(image, f"Supposed to be:{CorrectTopHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    i=0
    for k,v in BottomHalf.items():
        try:
            if(v[0]==CorrectBottomHalf[i]):
                color = (0, 255, 0)
                x1, y1, x2, y2 = map(int, v[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{CorrectBottomHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                color = (0, 0, 255)
                x1, y1, x2, y2 = map(int, v[1])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Supposed to be:{CorrectBottomHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            i=i+1
        except:
            color = (0, 0, 255)
            x1, y1, x2, y2 = map(int, v[1])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(image, f"Supposed to be:{CorrectBottomHalf[i]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite('static/images/marked_image.jpg', image)
    return render_template('Verified.html')



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
        
        update('static/images/cropped.jpg',correct_values)
        return render_template('display.html', values=correct_values)
    crop('static/images/uploaded_image.jpg')
    return render_template('correct_values.html', red_boxes=analyse('static/images/cropped.jpg'), enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
