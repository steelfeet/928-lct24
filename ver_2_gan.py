import argparse, subprocess
import os, sys, random, shutil, base64
from ultralytics import YOLO
import cv2

import time

from PIL import Image, ImageDraw
import zipfile


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

# директория файла
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
# куда рспаковываются изображения
UPLOAD_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)
OUT_DIR = os.path.join(BASE_DIR, "labels")
os.makedirs(OUT_DIR, exist_ok=True)


IM_SIZE = 512

ValidTransform = A.Compose([
    A.Resize(IM_SIZE,IM_SIZE),
    A.CenterCrop(IM_SIZE,IM_SIZE),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()])

# перевод YOLO в XY
def yolo2xy(yolo_x, yolo_y, yolo_width, yolo_height, width, height):
    center_x = yolo_x * width
    obj_width = yolo_width * width
    x1, x2 = center_x-obj_width/2, center_x+obj_width/2

    center_y = yolo_y * height
    obj_height = yolo_height * height
    y1, y2 = center_y-obj_height/2, center_y+obj_height/2
    return x1, x2, y1, y2

# перевод XY в YOLO
def xy2yolo(x1, y1, x2, y2, w, h):
    dw = 1./w
    dh = 1./h

    center_x = (x1 + x2)/2.0
    center_y = (y1 + y2)/2.0
    w = x2 - x1
    h = y2 - y1
    center_x = center_x * dw
    w = w * dw
    center_y = center_y * dh
    h = h * dh
    return center_x, center_y, w, h


def make_predictions(conf, iou, model, upload_filename, crop_args=None):
    if crop_args != None:
        offset_x, offset_y, crop_w, crop_h, full_w, full_h, full_path = crop_args

    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    predictions = model.predict(source=upload_path, conf=conf, iou=iou)
    predictions_list = []
    for idx, prediction in enumerate(predictions[0].boxes.xywhn):
        cls = int(predictions[0].boxes.cls[idx].item())
        # это Yolo-формат для обрезанного изображения
        center_x = prediction[0].item()
        center_y = prediction[1].item()
        yolo_width = prediction[2].item()
        yolo_height = prediction[3].item()

        if crop_args != None:
            image = Image.open(upload_path)
            draw = ImageDraw.Draw(image)
            # это XY-формат для обрезанного изображения
            xmin, xmax, ymin, ymax = yolo2xy(center_x, center_y, yolo_width, yolo_height, crop_w, crop_h)
            draw.rectangle(((xmin, ymin), (xmax,ymax)), fill="green")
            rected_filename = f"rect{idx}-{upload_filename}"
            rected_path = os.path.join(UPLOAD_DIR, rected_filename)
            image.save(rected_path, "JPEG")
            
            # это XY-формат для полного изображения
            xmin_full = xmin + offset_x
            xmax_full = xmax + offset_x
            ymin_full = ymin + offset_y
            ymax_full = ymax + offset_y
            
            # это Yolo-формат для полного изображения
            center_x, center_y, yolo_width, yolo_height = xy2yolo(xmin_full, ymin_full, xmax_full, ymax_full, full_w, full_h)
            
            # проверим
            xmin, xmax, ymin, ymax = yolo2xy(center_x, center_y, yolo_width, yolo_height, full_w, full_h)
            image = Image.open(full_path)
            draw = ImageDraw.Draw(image)
            draw.rectangle(((xmin, ymin), (xmax,ymax)), fill="green")
            rected_filename = f"full{idx}-{upload_filename}"
            rected_path = os.path.join(UPLOAD_DIR, rected_filename)
            image.save(rected_path, "JPEG")

        pred_str = f"{cls} {center_x} {center_y} {yolo_width} {yolo_height}"
        predictions_list.append(pred_str)
    
    return predictions_list


def xywh2x(img, xywh):
    obj_im_height, obj_im_width, channels = img.shape
    center_x, center_y, width, height = xywh

    center_x = float(center_x)
    center_y = float(center_y)
    width = float(width)
    height = float(height)

    xmin = int((center_x - (width / 2)) * obj_im_width)
    xmax = int((center_x + (width / 2)) * obj_im_width)
    ymin = int((center_y - (height / 2)) * obj_im_height)
    ymax = int((center_y + (height / 2)) * obj_im_height)

    return xmin, xmax, ymin, ymax


def classify(model, img, xywh):
    
    xmin, xmax, ymin, ymax = xywh2x(img, xywh)

    obj_width = xmax - xmin
    obj_height = ymax - ymin
    
    crop_img = img[ymin:ymax, xmin:xmax]

    if (obj_width < 50) and (obj_height < 50):
        gan_path = os.path.join(BASE_DIR, "Real-ESRGAN", "inference_realesrgan.py")
        subprocess_list = []
        subprocess_list.append("python")
        subprocess_list.append(gan_path)
        
        crop_path = os.path.join(UPLOAD_DIR, "4gan.jpg")
        cv2.imwrite(crop_path, crop_img)

        subprocess_list.append("-i")
        subprocess_list.append(crop_path)
        subprocess_list.append("-o")

        ganned_path = os.path.join(UPLOAD_DIR, "ganned")
        subprocess_list.append(ganned_path)
        subprocess_list.append("--outscale")
        subprocess_list.append("3.5")

        subprocess.run(subprocess_list) 

        crop_img = cv2.imread(os.path.join(ganned_path, "4gan_out.jpg"))

    input = ValidTransform(image=crop_img)
    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input["image"].unsqueeze(0)
    input = input.to(device)
    # Get prediction
    with torch.no_grad():
        logits = model(input)
    """
    ps = torch.exp(logits)        
    _, top_class = ps.topk(1, dim=1)
    """
    top_class = logits.argmax(dim=1)
    predicted_class = int(top_class.cpu()[0])
    
    prob = max(F.softmax(logits, dim=1).tolist()[0])
    

    return predicted_class, prob



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='images', help='Input zip file')
    parser.add_argument('-o', '--output', type=str, default='labels', help='Output zip file')
    parser.add_argument('-conf', '--conf', type=str, default='0.25', help='YOLO conf')
    parser.add_argument('-iou', '--iou', type=str, default='0.6', help='YOLO IOU')

    args = parser.parse_args()


    # извлекаем изображения
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    with zipfile.ZipFile(os.path.join(BASE_DIR, f"{args.input}.zip"), 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)




    start_time = time.time()

    yolo_model = YOLO(f"yolov8x.pt").load(os.path.join(BASE_DIR, "yolov8x.pt"))
    mnet_model_big = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    #перенастраиваем модель под наши классы
    for param in mnet_model_big.parameters():
        param.requires_grad = False
    n_inputs = mnet_model_big.classifier[0].in_features
    last_layer = nn.Linear(n_inputs, 5)
    mnet_model_big.classifier = last_layer

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mnet_model_big = mnet_model_big.to(device)
    mnet_model_big = mnet_model_big.to(device)
    torch.cuda.empty_cache()
    print("device: ", device)
    print()

    # если есть - загружаем веса
    BEST_MODEL_PATH = os.path.join(BASE_DIR, f"ml03_best.pth")
    if os.path.isfile(BEST_MODEL_PATH):
        print("mnet ml03_best exist, load")
        mnet_model_big.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device(device)))


    mnet_model_ganned = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    #перенастраиваем модель под наши классы
    for param in mnet_model_ganned.parameters():
        param.requires_grad = False
    n_inputs = mnet_model_ganned.classifier[0].in_features
    last_layer = nn.Linear(n_inputs, 5)
    mnet_model_ganned.classifier = last_layer

    # если есть - загружаем веса
    BEST_MODEL_PATH = os.path.join(BASE_DIR, f"ml04_ganned_best.pth")
    if os.path.isfile(BEST_MODEL_PATH):
        print("mnet ml04_ganned_best exist, load")
        mnet_model_ganned.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device(device)))


    # перебираем изображения
    predictions = ""
    for filename in os.listdir(UPLOAD_DIR):
        upload_path = os.path.join(UPLOAD_DIR, filename)
        basename, file_extension = os.path.splitext(filename)

        if file_extension == ".jpg" or file_extension == ".png":
            label_txt = ""
            classed_pred = []

            # смотрим размер
            img = cv2.imread(upload_path)
            full_h, full_w, _ = img.shape

            if full_h > 900 or full_w > 900:
                # надо кропать
                print("cropped")
                chunk_n_x = int(full_w / 640) + 1
                chunk_n_y = int(full_h / 640) + 1

                # режем не по 640 и сколько осталось, а по середине
                crop_w = int(full_w / chunk_n_x)
                crop_h = int(full_h / chunk_n_y)

                for chunk_x in range(0, chunk_n_x):
                    for chunk_y in range(0, chunk_n_y):
                        classed_pred = []
                        
                        offset_x = chunk_x*crop_w
                        offset_y = chunk_y*crop_h
                        imgCrop = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
                        crop_upload_filename = f"{basename}-{chunk_x}-{chunk_y}{file_extension}"
                        # сохраняем файл
                        crop_upload_path = os.path.join(UPLOAD_DIR, crop_upload_filename)
                        cv2.imwrite(crop_upload_path, imgCrop)

                        crop_args = offset_x, offset_y, crop_w, crop_h, full_w, full_h, upload_path

                        predictions = make_predictions(float(args.conf), float(args.iou), yolo_model, crop_upload_filename, crop_args)
                        
                        #распознаем класс найденного
                        for prediction in predictions:
                            val = str(prediction).split(" ")
                            obj_cls, xywh = val[0], val[1:5]

                            xmin, xmax, ymin, ymax = xywh2x(img, xywh)
                            obj_width = xmax - xmin
                            obj_height = ymax - ymin
                            
                            if (obj_cls == "4") or (obj_cls == "33") or (obj_cls == "34"):
                                if (obj_width > 50) and (obj_height > 50):
                                    cls, prob = classify(mnet_model_big, img, xywh)
                                else:
                                    cls, prob = classify(mnet_model_big, img, xywh)
                                pred_str = f"{cls} {val[1]} {val[2]} {val[3]} {val[4]}"
                                classed_pred.append(pred_str)

                            elif (obj_cls == "14"):
                                cls = 3          
                                pred_str = f"{cls} {val[1]} {val[2]} {val[3]} {val[4]}"
                                classed_pred.append(pred_str)

                                pred_str = f"{cls} {val[1]} {val[2]} {val[3]} {val[4]}"
                                classed_pred.append(pred_str)
                        
                        for prediction in classed_pred:
                            label_txt += f"{prediction}\n"

            #распознаем класс найденного
            predictions = make_predictions(float(args.conf), float(args.iou), yolo_model, filename)
            for prediction in predictions:
                val = str(prediction).split(" ")
                obj_cls, xywh = val[0], val[1:5]

                xmin, xmax, ymin, ymax = xywh2x(img, xywh)
                obj_width = xmax - xmin
                obj_height = ymax - ymin
                
                if (obj_cls == "4") or (obj_cls == "33") or (obj_cls == "34"):
                    if (obj_width > 50) and (obj_height > 50):
                        cls, prob = classify(mnet_model_big, img, xywh)
                    else:
                        cls, prob = classify(mnet_model_big, img, xywh)
                    
                    pred_str = f"{cls} {val[1]} {val[2]} {val[3]} {val[4]}"
                    classed_pred.append(pred_str)

                elif (obj_cls == "14"):
                    cls = 3
                    pred_str = f"{cls} {val[1]} {val[2]} {val[3]} {val[4]}"
                    classed_pred.append(pred_str)

            for prediction in classed_pred:
                label_txt += f"{prediction}\n"
            
            
            label_filename = f"{basename}.txt"
            label_path = os.path.join(OUT_DIR, label_filename)
            with open(label_path, 'w') as f:
                f.write(label_txt)

    end_time = time.time()
    shutil.make_archive(os.path.join(BASE_DIR, args.output), 'zip', OUT_DIR)


    print()
    print("success")
    print("Время работы в сек.: " + str(end_time - start_time))














