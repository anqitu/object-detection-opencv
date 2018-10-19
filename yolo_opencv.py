import cv2
import argparse
import numpy as np
from imutils import paths
import imutils
import csv
import os
import pandas as pd


# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image_dir', required=True,
#                 help = 'path to input directory of images')
# args = ap.parse_args()

# image_dir = args.image_dir

image_dir = 'images'
csv_path = 'result.csv'
image_paths = list(paths.list_images(image_dir))
config = 'yolov3.cfg'
weights = 'yolov3.weights'
classes = 'yolov3.txt'
COLOR = (128,128, 0)  #B, G, R



def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), COLOR, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

# pad images to a square
def crop_image(image):
    (h, w) = image.shape[:2]

    w_crop = int(w / 3)
    width = height = max(h, w_crop)
    padW = int((width - w_crop) / 2.0)
    padH = int((height - h) / 2.0)
    crop_images = []

    for i in range(0, w, w_crop):
        img = image[0:h, i:i+w_crop]
        img = cv2.copyMakeBorder(img, padH, padH, padW, padW, cv2.BORDER_CONSTANT)
        crop_images.append(img)

    return crop_images


def detect_cropped_image(image, classes, net):

    Height, Width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=True)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 0:
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    people_count = len(indices)

    return (image, people_count)

def detect_complete_image(image):

    images = []
    people_count_total = 0
    for image_crop in crop_image(image):
        (image, people_count) = detect_cropped_image(image_crop, classes, net)
        images.append(image)
        people_count_total += people_count
    image = np.concatenate(images, axis=1)

    Height, Width = image.shape[:2]
    cv2.putText(image, str(people_count_total), (int(Width * 0.05), int(Height * (1-0.05))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
    cv2.imwrite(image_path.replace(image_dir, 'detect'), image)

    return (image, people_count_total)

def get_classes():
    with open(classes, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_detected_image_paths(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w') as csv_file:
            fieldnames = ['image_path', 'people_count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.close()

    df = pd.read_csv(csv_path)
    return list(df['image_path'])

def write_to_csv(image_path, people_count):
    with open(csv_path, mode='a') as csv_file:
        fieldnames = ['image_path', 'people_count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'image_path': image_path, 'people_count': people_count})
        csv_file.close()


# writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
# writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})

classes = get_classes()
scale = 0.00392
net = cv2.dnn.readNet(weights, config)

detected_image_paths = get_detected_image_paths(csv_path)

for image_path in image_paths:
    (image, people_count_total) = detect_complete_image(cv2.imread(image_path))
    write_to_csv(image_path, people_count_total)
    print('{}: {}'.format(image_path, people_count_total))
