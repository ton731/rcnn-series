# %% [markdown]
# https://medium.com/codex/implementing-r-cnn-object-detection-on-voc2012-with-pytorch-b05d3c623afe

# %%

import urllib
import tarfile
from pathlib import Path
import os
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, models, transforms
from tqdm import tqdm
import pickle
from PIL import Image
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from datetime import datetime
import torchmetrics
import logging

# %% [markdown]
# Hyperparameters configuration

# %%
config = {'image_size':224, 'n_classes':21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals':2000, 'pad': 16, 'confidence_threshold': 0.95}
train_config = {'epochs': 5, 'batch_size':32, 'lr': 0.001, 'lr_decay':0.5, 'l2_reg': 1e-5, 'ckpt_dir': Path('results')}
load_path = None
# load_path = Path("results/2023_01_10__11_12_25")
voc_2012_classes = ['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

# %% [markdown]
# Download data

# %%
class VOCDataset:
    def __init__(self):
        self.root = Path("data")
        self.root.mkdir(parents=True, exist_ok=True)
        self.train_dir = None
        self.test_dir = None
        self.train_data_link = None
        self.test_data_link = None


    def common_init(self):
        # init for shared subclasses
        self.label_type = ['none','aeroplane',"Bicycle",'bird',"Boat","Bottle","Bus","Car","Cat","Chair",'cow',"Diningtable","Dog","Horse","Motorbike",'person', "Pottedplant",'sheep',"Sofa","Train","TVmonitor"]
        self.convert_id = ['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Dining table","Dog","Horse","Motorbike",'Person', "Potted plant",'Sheep',"Sofa","Train","TV/monitor"]
        self.convert_labels = {}
        for i, x in enumerate(self.label_type):
            self.convert_labels[x.lower()] = i

        self.num_classes = len(self.label_type)     # 20 class + 1 background
    

    def download_dataset(self, validation_size=5000):
        # download voc train dataset
        if os.path.exists(self.root / "voctrain.tar"):
            print("[*] Dataset already exists!")
        else:
            print("[*] Downloading dataset...")
            print(self.train_data_link)
            urllib.request.urlretrieve(self.train_data_link, self.root / "voctrain.tar")

        if os.path.exists(self.root / "VOCtrain"):
            print("[*] Dataset is already extracted!")
        else:
            print("[*] Extracting dataset...")
            tar = tarfile.open(self.root / "voctrain.tar")
            tar.extractall(self.root / "VOCtrain")
            tar.close()

        # download test dataset
        if os.path.exists(self.root / "VOCtest"):
            print("[*] Test dataset already exist!")
        else:
            if self.test_data_link is None:
                # move 5k images to validation set
                print("[*] Moving validation data...")
                test_annotation_dir = self.test_dir / "Annotations"
                test_img_dir = self.test_dir / "JPEGImages"
                test_annotation_dir.mkdir(parents=True, exist_ok=True)
                test_img_dir.mkdir(parents=True, exist_ok=True)

                random.seed(731)
                val_img_paths = random.sample(sorted(os.listdir(self.train_dir / "JPEGImages")), validation_size)

                for path in val_img_paths:
                    img_name = str(path).split("/")[-1].split(".")[0]
                    # move image
                    os.rename(self.train_dir / "JPEGImages" / f"{img_name}.jpg", test_img_dir / f"{img_name}.jpg")
                    # move annotation
                    os.rename(self.train_dir / "Annotations" / f"{img_name}.xml", test_annotation_dir / f"{img_name}.xml")
            else:
                # load from val data
                print("[*] Downloading validation dataset...")
                urllib.request.urlretrieve(self.test_data_link, "voctset.tar")

                print("[*] Extracting validation dataset...")
                tar = tarfile.open("voctest.tar", "r:")
                tar.extractall("/content/VOCtest")
                tar.close()
                # os.remove("/content/voctset.tar")


    def read_xml(self, xml_path):
        object_list = []

        tree = ET.parse(open(xml_path, 'r'))
        root = tree.getroot()

        objects = root.findall("object")
        for _object in objects:
            name = _object.find("name").text
            bndbox = _object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_name = _object.find("name").text
            object_list.append({'x1':xmin, 'x2':xmax, 'y1':ymin, 'y2':ymax, 'class':self.convert_labels[class_name]})
        
        return object_list


# %%
class VOC2007(VOCDataset):
    def __init__(self):
        super().__init__()
        self.train_dir = self.root / 'VOCtrain/VOCdevkit/VOC2007'
        self.test_dir = self.root / 'VOCtest/VOCdevkit/VOC2007'
        self.train_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
        self.test_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
        self.common_init()  #mandatory
    
class VOC2012(VOCDataset):
    def __init__(self):
        super().__init__()
        self.train_dir = self.root / 'VOCtrain/VOCdevkit/VOC2012'
        self.test_dir = self.root / 'VOCtest/VOCdevkit/VOC2012'
        # original site goes down frequently, so we use a link to the clone alternatively
        # self.train_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' 
        self.train_data_link = 'http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
        self.test_data_link = None
        self.common_init()  #mandatory

# %%
voc_dataset = VOC2012()
voc_dataset.download_dataset()

# %%
train_data_num = len(os.listdir(voc_dataset.train_dir / "Annotations"))
valid_data_num = len(os.listdir(voc_dataset.test_dir / "Annotations"))
print("train data num:", train_data_num)
print("valid data num:", valid_data_num)

# %% [markdown]
# Show some data

# %%
def plot_img(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def plot_img_from_path(img_path):
    print("plotting:", img_path)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("img shape:", img_rgb.shape)
    plt.imshow(img_rgb)
    plt.show()
    
    
def plot_bounding_box_from_path(img_path):
    # img
    print("plotting:", img_path)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    tree = ET.parse(open(xml_path, 'r'))
    root = tree.getroot()
    
    # print image shape
    w, h = root.find("size").find("width").text, root.find("size").find("height").text
    print("width, height:", w, h)

    # bounding box settings
    box_img = img_bgr.copy()
    bbox_color = (0, 69, 255)    # bgr
    bbox_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = bbox_color
    line_type = 1

    # plot box
    objects = root.findall("object")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        class_name = _object.find('name').text

        cv2.rectangle(box_img, (xmin, ymin), (xmax, ymax), bbox_color, bbox_thickness)
        cv2.putText(box_img, class_name, (xmin, ymin-5), font, font_scale, font_color, line_type)

    box_img_rgb = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
    result = np.hstack((img_rgb, box_img_rgb))
    # plt.imshow(result)
    plt.imshow(img_rgb)
    plt.show()
    plt.imshow(box_img_rgb)
    plt.show()

# %%
'''
img_name = random.choice(os.listdir(voc_dataset.train_dir / "JPEGImages"))
img_path = str(voc_dataset.train_dir / "JPEGImages" / img_name)
plot_bounding_box_from_path(img_path)
'''

# %% [markdown]
# Helper functions

# %%
def selective_search(img):
    # return region proposals of selective searh over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects


def plot_results(img, bboxes, color=(0, 69, 255)):
    plot_cfg = {'bbox_color':color, 'bbox_thickness':2, 
                'font':cv2.FONT_HERSHEY_SIMPLEX, 'font_scale':0.5, 'font_color':color, 'line_thickness':1}
    img_bb = img.copy()
    for box in bboxes:
        bbox = box['bbox']
        cv2.rectangle(img_bb, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), plot_cfg['bbox_color'], plot_cfg['bbox_thickness'])
        cv2.putText(img_bb, f"{voc_dataset.label_type[box['class']]}, {str(box['conf'])[:5]}",  (bbox['x1'], bbox['y1'] - 5), plot_cfg['font'], 
                    plot_cfg['font_scale'], plot_cfg['font_color'], plot_cfg['line_thickness'])
    return img_bb

# %%
'''
img_path = str(voc_dataset.train_dir / "JPEGImages" / img_name)
img = cv2.imread(img_path)
rects = selective_search(img)
for i, rect in enumerate(rects):
    if i > 2000:
        break
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 255, 100), 1)
plot_img(img)
'''

# %% [markdown]
# IoU (Intersection over Union)

# %%
def calculate_IoU(bb1, bb2):
    # IoU = area_of_overlap / area_of_union
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    
    # return IoU as 0 if 2 boxes are not intersected
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    union_area = bb1_area + bb2_area - intersection_area

    # iou
    iou = intersection_area / union_area
    return iou
    

# %%
bb1 = {'x1':100, 'x2':200, 'y1':50, 'y2':150}
bb2 = {'x1':150, 'x2':300, 'y1':120, 'y2':200}
print("my iou:", calculate_IoU(bb1, bb2))

# %% [markdown]
# mAP (Mean Average Precision)

# %%
def calculate_mAP(pred, truth, iou_threshold=0.5, num_classes=21, per_class=False):
    pass

# %% [markdown]
# NMS (Non-max suppression)

# %%
def nms(bboxes, iou_threshold=0.5):
    # bboxes: list of dicts {'bbox':(x1,x2,y1,y2), 'confidence':float, 'class':int}
    confidence_list = np.array([bbox['confidence'] for bbox in bboxes])
    confidence_order = (-confidence_list).argsort()   # apply minus to make the order descending
    is_removed = [False for _ in range(len(bboxes))]
    selected_bboxes = []
    
    for i in range(len(bboxes)):
        bbox_idx = confidence_order[i]
        if is_removed[bbox_idx]:
            continue
        
        # add bbox to the main bbox object
        selected_bboxes.append(bboxes[bbox_idx])
        is_removed[bbox_idx] = True
        
        # remove overlapping bboxes
        for order in range(i+1, len(bboxes)):
            other_bbox_idx = confidence_order[order]
            # check if the bbox not remove yet
            if is_removed[other_bbox_idx] == False:
                # check overlapping
                iou = calculate_IoU(bboxes[bbox_idx]['bbox'], bboxes[other_bbox_idx]['bbox'])
                if iou >= iou_threshold:
                    is_removed[other_bbox_idx] = True
    
    return selected_bboxes
        
        
    

# %% [markdown]
# Datasets and Dataloaders

# %% [markdown]
# 1. Domain-specific fine tuning

# %%
class RCNN_Dataset(torch.utils.data.Dataset):
    """
    It's for domain-specific fine-tuning, inputs are the cropped image of the bounding
    boxes, and outputs are the labels of the cropped images, such as background, class 1,
    class 2, ... class N.
    """
    def __init__(self, dataset, cfg, IoU_threshold={'positive':0.5, 'partial':0.3},
                 sample_ratio={'object':32, 'background':96}, data_path=Path("data/domain-specific")):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((cfg['image_size'], cfg['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if self.dataset_exists() == False:
            self.generate_dataset(sample_ratio, IoU_threshold)
        else:
            print("[*] Loading domain-specific dataset from", self.data_path)
            with open(self.data_path / "train_images.pkl", 'rb') as f:
                self.train_images = pickle.load(f)
            with open(self.data_path / "train_labels.pkl", 'rb') as f:
                self.train_labels = pickle.load(f)
                
            # check if both files are complete and flawless
            if not len(self.train_images) == len(self.train_labels):
                raise ValueError("The loaded dataset is invalid or in different size.")
    
    
    def __len__(self):
        return len(self.train_labels)
    

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        img = Image.fromarray(cv2.cvtColor(self.train_images[i], cv2.COLOR_BGR2RGB))
        return {'image': self.transform(img), 'label': self.train_labels[i][0],
                'proposed_bbox': self.train_labels[i][1], 'gt_bbox': self.train_labels[i][2]}
        
        
    def dataset_exists(self):
        if os.path.exists(self.data_path / "train_images.pkl") == False:
            return False
        if os.path.exists(self.data_path / "train_labels.pkl") == False:
            return False
        return True
    
    
    def generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
        print("[*] Generating dataset for R-CNN")
        image_dir = self.dataset.train_dir / "JPEGImages"
        annot_dir = self.dataset.train_dir / "Annotations"
        object_counter = 0
        background_counter = 0
        self.train_images = []
        self.train_labels = []

        pbar = tqdm(sorted(os.listdir(image_dir))[:2000], position=0, leave=True)
        for img_name in pbar:
            pbar.set_description(f"Data size: {len(self.train_labels)}")
            
            # load image and ground truth bounding boxes
            img = cv2.imread(str(image_dir / img_name))
            xml_path = annot_dir / img_name.replace(".jpg", ".xml")
            gt_bboxes = self.dataset.read_xml(xml_path)
            
            # generate bounding box proposals from selective search
            rects = selective_search(img)[:2000]  # use only 2000 box
            random.shuffle(rects)
            
            # loop through all proposals
            for (x, y, w, h) in rects:
                # apply padding
                x1, x2 = np.clip([x - padding, x + w + padding], 0, img.shape[1])
                y1, y2 = np.clip([y - padding, y + h + padding], 0, img.shape[0])
                proposed_bbox = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                
                # check the proposal with every elements of the ground truth boxes
                is_object = False   # define flag
                for gt_bbox in gt_bboxes:
                    iou = calculate_IoU(gt_bbox, proposed_bbox)
                    if iou >= IoU_threshold['positive']:    # iou >= 0.5
                        object_counter += 1
                        # add image
                        cropped = img[y1 : y2, x1 : x2, :]
                        self.train_images.append(cropped)
                        # add label, here x, y is the center of the rectangle
                        proposed_bbox_xywh = ((x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1))
                        gt_bbox_xywh = ((gt_bbox['x1']+gt_bbox['x2'])/2, (gt_bbox['y1']+gt_bbox['y2'])/2, 
                                        (gt_bbox['x2']-gt_bbox['x1']), (gt_bbox['y2']-gt_bbox['y1']))
                        self.train_labels.append([gt_bbox['class'], proposed_bbox_xywh, gt_bbox_xywh])
                        
                        is_object = True
                        break
                
                # if the proposal is not close to any ground truth box
                if background_counter < sample_ratio['background'] and is_object == False:
                    background_counter += 1
                    # add background image
                    cropped = img[y1 : y2, x1 : x2, :]
                    self.train_images.append(cropped)
                    # add background label
                    proposed_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                    gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                    self.train_labels.append([0, proposed_bbox_xywh, gt_bbox_xywh])
                
                # control the ratio between object and backgruond
                if object_counter >= sample_ratio['object'] and background_counter == sample_ratio['background']:
                    object_counter -= sample_ratio['object']
                    background_counter = 0
        
        print("[*] Dataset generated! Saving labels to ", self.data_path)
        with open(self.data_path / "train_images.pkl", "wb") as f:
            pickle.dump(self.train_images, f)
        with open(self.data_path / "train_labels.pkl", "wb") as f:
            pickle.dump(self.train_labels, f)

# %%
def RCNN_DataLoader(voc_dataset, cfg, train_cfg, shuffle=True):
    rcnn_dataset = RCNN_Dataset(voc_dataset, cfg)
    return torch.utils.data.DataLoader(rcnn_dataset, batch_size=train_cfg['batch_size'], shuffle=shuffle)

# %%
rcnn_loader = RCNN_DataLoader(voc_dataset, config, train_config)

# %%
'''
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    transforms.ToPILImage()
])

for x in rcnn_loader:
    for i in range(5):
        plt.imshow(inverse_transform(x['image'][i]))
        plt.title(voc_2012_classes[x['label'][i]])
        plt.show()
    break
'''

# %% [markdown]
# 2. Object category classification

# %%
class RCNN_Classification_Dataset(torch.utils.data.Dataset):
    """
    It's for the final object category classification model. The positive samples
    are from the ground truth bboxes, and the negative samples are from the bbox
    proposed with IoU smaller than 0.3.
    """
    def __init__(self, dataset, cfg, IoU_threshold={'positive':0.5, 'partial':0.3},
                 sample_ratio={'object':32, 'background':96}, data_path=Path("data/object-classification")):
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((cfg['image_size'], cfg['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if self.dataset_exists() == False:
            self.generate_dataset(sample_ratio, IoU_threshold)
        else:
            print("[*] Loading object-classification dataset from", self.data_path)
            with open(self.data_path / "train_images_classification.pkl", 'rb') as f:
                self.train_images = pickle.load(f)
            with open(self.data_path / "train_labels_classification.pkl", 'rb') as f:
                self.train_labels = pickle.load(f)
                
            # check if both files are complete and flawless
            if not len(self.train_images) == len(self.train_labels):
                raise ValueError("The loaded dataset is invalid or in different size.")
    
    
    def __len__(self):
        return len(self.train_labels)
    

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        img = Image.fromarray(cv2.cvtColor(self.train_images[i], cv2.COLOR_BGR2RGB))
        return {'image': self.transform(img), 'label': self.train_labels[i][0],
                'proposed_bbox': self.train_labels[i][1], 'gt_bbox': self.train_labels[i][2]}
        
        
    def dataset_exists(self):
        if os.path.exists(self.data_path / "train_images_classification.pkl") == False:
            return False
        if os.path.exists(self.data_path / "train_labels_classification.pkl") == False:
            return False
        return True
    
    
    def generate_dataset(self, sample_ratio, IoU_threshold, padding=16):
        print("[*] Generating dataset for R-CNN (object classification)")
        image_dir = self.dataset.train_dir / "JPEGImages"
        annot_dir = self.dataset.train_dir / "Annotations"
        object_counter = 0
        self.train_images = []
        self.train_labels = []

        pbar = tqdm(sorted(os.listdir(image_dir)), position=0, leave=True)
        for img_name in pbar:
            pbar.set_description(f"Data size: {len(self.train_labels)}")
            
            # load image and ground truth bounding boxes
            img = cv2.imread(str(image_dir / img_name))
            xml_path = annot_dir / img_name.replace(".jpg", ".xml")
            gt_bboxes = self.dataset.read_xml(xml_path)
            
            # directly use ground truth bboxes as positive samples
            for gt_bbox in gt_bboxes:
                cropped = img[gt_bbox['y1']:gt_bbox['y2'], gt_bbox['x1']:gt_bbox['x2'], :]
                self.train_images.append(cropped)
                gt_bbox_xywh = ((gt_bbox['x1']+gt_bbox['x2'])/2, (gt_bbox['y1']+gt_bbox['y2'])/2, 
                                (gt_bbox['x2']-gt_bbox['x1']), (gt_bbox['y2']-gt_bbox['y1']))
                self.train_labels.append([gt_bbox['class'], gt_bbox_xywh, gt_bbox_xywh])
            object_counter += len(gt_bboxes)  
            
            # collect background
            if object_counter >= sample_ratio['object']:
                object_counter -= sample_ratio['object']
                background_counter = 0
                
                # generate bbox proposals with selective search
                rects = selective_search(img)[:2000]  
                random.shuffle(rects)
                   
                # loop through all proposals
                for (x, y, w, h) in rects:
                    # apply padding
                    x1, x2 = np.clip([x - padding, x + w + padding], 0, img.shape[1])
                    y1, y2 = np.clip([y - padding, y + h + padding], 0, img.shape[0])
                    proposed_bbox = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                    is_object = False
                    
                    # check the proposal with every elements of the ground truth boxes
                    for gt_bbox in gt_bboxes:
                        iou = calculate_IoU(gt_bbox, proposed_bbox)
                        if iou >= IoU_threshold['partial']:    # if object
                            is_object = True
                            break
                        
                    # save proposal if it's background
                    if is_object == False:
                        background_counter += 1
                        cropped = img[y1 : y2, x1 : x2, :]
                        self.train_images.append(cropped)
                        proposed_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                        gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                        self.train_labels.append([0, proposed_bbox_xywh, gt_bbox_xywh])
                      
                    # control the ration of object and background
                    if background_counter == sample_ratio['background']:
                        break
        
        print("[*] Dataset generated! Saving labels to ", self.data_path)
        with open(self.data_path / "train_images_classification.pkl", "wb") as f:
            pickle.dump(self.train_images, f)
        with open(self.data_path / "train_labels_classification.pkl", "wb") as f:
            pickle.dump(self.train_labels, f)

# %%
def RCNN_Classificaion_DataLoader(voc_dataset, cfg, train_cfg, shuffle=True):
    rcnn_classification_dataset = RCNN_Classification_Dataset(voc_dataset, cfg)
    return torch.utils.data.DataLoader(rcnn_classification_dataset, batch_size=train_cfg["batch_size"], shuffle=shuffle)

# %%
rcnn_classification_loader = RCNN_Classificaion_DataLoader(voc_dataset, config, train_config)

# %% [markdown]
# Model

# %%
class RCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg['n_classes']
        self.do_bbox_reg = cfg['bbox_reg']
        self.max_proposals = cfg['max_proposals']
        self.img_size = cfg['image_size']
        self.confidence_threshold = cfg['confidence_threshold']

        self._initialize_weights()


    def _initialize_weights(self):
        print("[*] Initializing new network...")
        self.effnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=self.num_classes)
        self.convnet = self.effnet.extract_features
        self.flatten = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, self.num_classes)
        )
        self.softmax = nn.Softmax()
        if self.do_bbox_reg:
            self.bbox_reg = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(1280, 4)
            )


    def refine_bbox(self, bbox, pred):
        # bbox is list of [{x1, x2, y1, y2}, ...]
        # pred is a sizr 4 array of predicted refinement of shape
        x, y = (bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2
        w, h = bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']

        new_x = x + w * pred[0]
        new_y = y + h * pred[1]
        new_w = w * torch.exp(pred[2])
        new_h = h * torch.exp(pred[3])

        return {'x1': new_x - new_w/2, 'x2': new_x + new_w/2, 'y1': new_y - new_h/2, 'y2': new_y + new_h/2}


    def inference_single(self, img, rgb=False, batch_size=8, apply_nms=True, nms_threshold=0.2):
        # img have to be loaded in BGR colorspace, or else rgb should be True
        self.eval()
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # perform selective search to find region proposals
        rects = selective_search(img)
        proposals = []
        boxes = []
        for (x, y, w, h) in rects[:self.max_proposals]:
            roi = cv2.cvtColor(img[y:y+h, x:x+w, :], cv2.COLOR_RGB2BGR)
            roi = preprocess(roi)
            proposals.append(roi)
            boxes.append({'x1':x, 'y1':y, 'x2':x+w, 'y2':y+h})
        
        # convert to dataloader for batching
        proposals = torch.stack(proposals)
        proposals = torch.Tensor(proposals)
        proposals = torch.utils.data.TensorDataset(proposals)
        proposals = torch.utils.data.DataLoader(proposals, batch_size=batch_size)

        # predict probability of each box
        bacth_iter = 0
        useful_bboxes = []
        for proposal_batch in tqdm(proposals):
            patches = proposal_batch[0].to(device)

            with torch.no_grad():
                features = self.convnet(patches)
                features = self.flatten(features)
                pred = self.classifier(features)
                pred = self.softmax(pred)

                if self.do_bbox_reg:
                    bbox_refine = self.bbox_reg(features)
            
            # patches which are not classsified as background
            useful_idx = torch.where(pred.argmax(1) > 0)    
            for i in useful_idx[0]:
                # loop through all useful patches
                i = i.cpu().detach().numpy()
                estimate = {}

                class_prob = pred[i].cpu().detach().numpy()
                estimate['class'] = class_prob.argmax(0)
                estimate['confidence'] = class_prob.max(0)

                original_bbox = boxes[bacth_iter * batch_size + i]
                if self.do_bbox_reg == False:
                    estimate['bbox'] = original_bbox
                else:
                    estimate['bbox'] = self.refine_bbox(original_bbox, bbox_refine[i])
                
                if estimate['confidence'] > self.confidence_threshold:
                    useful_bboxes.append(estimate)

            bacth_iter += 1

        # apply non-max suppression to remove duplicate boxes
        if apply_nms:
            useful_bboxes = nms(useful_bboxes, nms_threshold)

        # convert bboxes values to numpy
        for i in range(len(useful_bboxes)):
            useful_bboxes[i]["bbox"]["x1"] = int(useful_bboxes[i]["bbox"]["x1"].cpu().numpy())
            useful_bboxes[i]["bbox"]["y1"] = int(useful_bboxes[i]["bbox"]["y1"].cpu().numpy())
            useful_bboxes[i]["bbox"]["x2"] = int(useful_bboxes[i]["bbox"]["x2"].cpu().numpy())
            useful_bboxes[i]["bbox"]["y2"] = int(useful_bboxes[i]["bbox"]["y2"].cpu().numpy())

        
        return useful_bboxes
    

    def inference(self, imgs, rgb=False, batch_size=8, apply_nms=True, nms_threshold=0.2):
        # when given single image
        if type(imgs) == np.ndarray and len(imgs.shape) == 3:
            return self.inference_single(imgs, rgb, batch_size, apply_nms)

        bboxes = []
        for img in tqdm(imgs):
            pred_bboxes = self.inference_single(img, rgb, batch_size, apply_nms)
            bboxes.append(pred_bboxes)
        return bboxes

# %% [markdown]
# Trainer

# %%
class RCNN_Trainer:
	def __init__(self, model, fine_tuning_loader, category_classification_loader, train_cfg):
		self.model = model
		self.fine_tuning_loader = fine_tuning_loader
		self.category_classification_loader = category_classification_loader
		self.train_cfg = train_cfg

		self._init_env()
		
	
	def _init_env(self):
		# checkpoint directory
		ckpt_dir = self.train_cfg['ckpt_dir']
		ckpt_dir = ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
		ckpt_dir.mkdir(parents=True, exist_ok=True)
		self.ckpt_dir = ckpt_dir
	 
		# logger
		logger = logging.getLogger(name='RCNN')
		logger.setLevel(level=logging.INFO)
		# set formatter
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		# console handler
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)
		# file handler
		file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		
		self.logger = logger
		self.logger.info(self.train_cfg)
	
	
	def fine_tune_training(self):
		# define loss function
		criterion_ce = nn.CrossEntropyLoss()
		criterion_mse = nn.MSELoss()
		
		# compute the gradients for the entire model
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg['lr'], weight_decay=self.train_cfg['l2_reg'])
		
		# lr scheduler
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.train_cfg['lr_decay'])
		
		# initialize metrics
		accuracy_counter = torchmetrics.Accuracy(task="multiclass", num_classes=21)
		
		# train loop
		best_train_loss = np.inf
		for epoch in range(self.train_cfg['epochs']):
			self.model.train()
			print(f"[*] Training fine-tuning epoch {epoch+1}/{self.train_cfg['epochs']}")
			pbar = tqdm(self.fine_tuning_loader)
			train_loss = 0
			train_acc = 0
			train_iter = 0
			for step, batch in enumerate(pbar):
				features = self.model.convnet(batch['image'].to(device))
				features = self.model.flatten(features)
				output = self.model.classifier(features)
				
				# loss
				clf_loss = criterion_ce(output, batch['label'].to(device))
				loss = clf_loss
				
				# bbox regression loss
				if self.model.do_bbox_reg:
					bbox_est = self.model.bbox_reg(features)
					p_x, p_y, p_w, p_h = batch['proposed_bbox'][0], batch['proposed_bbox'][1], batch['proposed_bbox'][2], batch['proposed_bbox'][3]
					g_x, g_y, g_w, g_h = batch['gt_bbox'][0], batch['gt_bbox'][1], batch['gt_bbox'][2], batch['gt_bbox'][3]

					bbox_ans = torch.stack([(g_x-p_x)/p_w, (g_y-p_y)/p_h, torch.log(g_w)/p_w, torch.log(g_h)/p_h], axis=1)
					bbox_ans = bbox_ans.float().to(device)
					
					# count only bboxes that are not backgrond
					not_bg = (batch['label']>0).reshape(len(batch['label']), 1).to(device)
					bbox_est = bbox_est * not_bg
					bbox_ans = bbox_ans * not_bg
					
					# add to loss
					bbox_loss = criterion_mse(bbox_est, bbox_ans)
					loss += bbox_loss

				# backpropagation
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				train_loss += loss.cpu().detach().numpy()
				
				# logging
				acc = accuracy_counter(output.cpu(), batch['label'])
				train_acc += acc.numpy()
				train_iter += 1
				pbar.set_description(f"loss: {loss.cpu().detach().numpy():3f},  accuracy: {acc.numpy():.3f}")
			self.logger.info(f"Epoch: {epoch+1}, loss: {train_loss:3f},  accuracy: {train_acc/train_iter:.3f}")
			
			# save model
			if train_loss < best_train_loss:
				best_train_loss = train_loss
				torch.save(self.model.state_dict(), self.ckpt_dir / "model.pt")
			
			# update lr
			scheduler.step()
		
	
	def category_classification_training(self):
		# define loss functionn
		criterion_ce = nn.CrossEntropyLoss()
		
		# gradiens for final classifier only
		optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.train_cfg['lr'], weight_decay=self.train_cfg['l2_reg'])
		
		# initialize metrics
		accuracy_counter = torchmetrics.Accuracy(task="multiclass", num_classes=21)
		
		# train loop
		best_train_loss = np.inf
		for epoch in range(self.train_cfg['epochs']):
			self.model.train()
			print(f"[*] Training category-classification epoch {epoch+1}/{self.train_cfg['epochs']}")
			pbar = tqdm(self.category_classification_loader)
			train_loss = 0
			train_acc = 0
			train_iter = 0
			for step, batch in enumerate(pbar):
				features = self.model.convnet(batch['image'].to(device))
				features = self.model.flatten(features)
				output = self.model.classifier(features)
				
				# backpropagation
				clf_loss = criterion_ce(output, batch['label'].to(device))
				loss = clf_loss
				train_loss += loss.cpu().detach().numpy()
				
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				
				# logging
				acc = accuracy_counter(output.cpu(), batch['label'])
				train_acc += acc.numpy()
				train_iter += 1
				pbar.set_description(f"loss: {loss.cpu().detach().numpy():3f},  accuracy: {acc.numpy():.3f}")
			self.logger.info(f"Epoch: {epoch+1}, loss: {train_loss:3f},  accuracy: {train_acc/train_iter:.3f}")
			
			# save model
			if train_loss < best_train_loss:
				best_train_loss = train_loss
				torch.save(self.model.state_dict(), self.ckpt_dir / "model.pt")

# %% [markdown]
# Train

# %%
model = RCNN(config).to(device)
if load_path:
    model.load_state_dict(torch.load(load_path / "model.pt"))
trainer = RCNN_Trainer(model, rcnn_loader, rcnn_classification_loader, train_config)

# %%
trainer.fine_tune_training()

# %%
trainer.category_classification_training()

# %% [markdown]
# Visualize prediction

# %%
@torch.no_grad()
def visualize(voc_dataset, model, img_path, i, save_to_disk=False):
    img = cv2.imread(img_path)
    pred_bboxes = model.inference(img, apply_nms=True, nms_threshold=0.2)
    pred_bboxes=nms(pred_bboxes, 0.2)
    # plot predicted
    # print("Predicted:")
    predcited_name = f"{i}_pred.png"
    plot_bboxes(img, pred_bboxes, predcited_name, save_to_disk)
    # plot ground truth
    xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
    gt_bboxes = read_gt_xml(xml_path)
    # print("Ground truth")
    gt_name = f"{i}_truth.png"
    plot_bboxes(img, gt_bboxes, gt_name, save_to_disk)


def read_gt_xml(xml_path):
    tree = ET.parse(open(xml_path, 'r'))

    root=tree.getroot()

    obj_list = []
    objects = root.findall("object")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        class_name = _object.find('name').text

        obj_list.append({'class': voc_dataset.convert_labels[class_name], 'confidence': 1.0, 
                         'bbox': {'x1':xmin, 'x2':xmax, 'y1':ymin, 'y2':ymax}})
    return obj_list



def plot_bboxes(img, bboxes, save_name, save_to_disk, color=(255, 69, 0)):
    plot_cfg = {'bbox_color':color, 'bbox_thickness':2, 
                'font':cv2.FONT_HERSHEY_SIMPLEX, 'fontScale':0.5, 'fontColor':color, 'lineThickness':1}
    img_bbox = img.copy()
    img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
    for box in bboxes:
        bbox = box['bbox']
        cv2.rectangle(img_bbox, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), plot_cfg['bbox_color'], plot_cfg['bbox_thickness'])
        cv2.putText(img_bbox, f"{voc_dataset.label_type[box['class']]}, {str(box['confidence'])[:5]}",  (bbox['x1'], bbox['y1'] - 5), plot_cfg['font'], 
                    plot_cfg['fontScale'], plot_cfg['fontColor'], plot_cfg['lineThickness'])
    plt.imshow(img_bbox)
    if save_to_disk:
        plt.savefig(trainer.save_result_dir / save_name)
    else:
        plt.show()
    plt.close()

# %%
# plot and save
trainer.save_result_dir = trainer.ckpt_dir / "predictions"
trainer.save_result_dir.mkdir(parents=True, exist_ok=True)
plot_num = 200
for i in range(plot_num):
    img_name = sorted(os.listdir(voc_dataset.test_dir / "JPEGImages"))[i]
    img_path = str(voc_dataset.test_dir / "JPEGImages" / img_name)
    print("plotting:", img_path)
    visualize(voc_dataset, model, img_path, i=i, save_to_disk=True)

# %%



