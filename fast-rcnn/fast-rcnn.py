# %% [markdown]
# reference: https://github.com/zjZSTU/Fast-R-CNN

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
from datetime import datetime
import torchmetrics
import logging
from torchvision.ops import roi_pool

# %% [markdown]
# Hyperparameters configuration

# %%
config = {'image_size':224, 'n_classes':21, 'bbox_reg': True, 'network': 'efficientnet-b0', 'max_proposals':2000, 'pad': 16, 'confidence_threshold': 0.95}
train_config = {'epochs': 1, 'batch_size':2, 'lr': 0.001, 'lr_decay':0.5, 'l2_reg': 1e-5, 'ckpt_dir': Path('results')}
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
        self.root = Path("../data")
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
# Helper functions

# %%
def selective_search(img):
    # return region proposals of selective searh over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

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
# Fast R-CNN Dataset

# %%
class Fast_RCNN_Dataset(torch.utils.data.Dataset):
    """
    It's for domain-specific fine-tuning, inputs are the cropped image of the bounding
    boxes, and outputs are the labels of the cropped images, such as background, class 1,
    class 2, ... class N.
    """
    def __init__(self, dataset, cfg, IoU_threshold={'positive':0.5, 'negative':0.1},
                 sample_ratio={'object':32, 'background':96}, data_path=Path("fast-rcnn_data/")):
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
            print("[*] Loading Fast-RCNN dataset from", self.data_path)
            with open(self.data_path / "train_rois.pkl", 'rb') as f:
                self.train_rois = pickle.load(f)
            with open(self.data_path / "train_labels.pkl", 'rb') as f:
                self.train_labels = pickle.load(f)
    
    
    def __len__(self):
        return len(self.train_rois)
    

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        img = Image.fromarray(cv2.cvtColor(self.train_images[i], cv2.COLOR_BGR2RGB))
        return {'image': self.transform(img), 'label': self.train_labels[i][0],
                'proposed_bbox': self.train_labels[i][1], 'gt_bbox': self.train_labels[i][2]}
        
        
    def dataset_exists(self):
        if os.path.exists(self.data_path / "train_rois.pkl") == False:
            return False
        if os.path.exists(self.data_path / "train_labels.pkl") == False:
            return False
        return True
    
    
    def generate_dataset(self, sample_ratio, IoU_threshold):
        print("[*] Generating dataset for Fast-RCNN")
        image_dir = self.dataset.train_dir / "JPEGImages"
        annot_dir = self.dataset.train_dir / "Annotations"
        self.train_rois = []
        self.train_labels = []

        pbar = tqdm(sorted(os.listdir(image_dir)), position=0, leave=True)
        for img_name in pbar:
            pbar.set_description(f"Data size: {len(self.train_rois)}")
            
            # load image and ground truth bounding boxes
            img = cv2.imread(str(image_dir / img_name))
            xml_path = annot_dir / img_name.replace(".jpg", ".xml")
            gt_bboxes = self.dataset.read_xml(xml_path)
            
            # generate bounding box proposals from selective search
            rects = selective_search(img)[:2000]  # use only 2000 box
            random.shuffle(rects)
            
            positive_rois = []
            negative_rois = []
            positive_labels = []
            negative_labels = []
            
            # loop through all proposals
            for (x, y, w, h) in rects:
                # get the 4 points
                x1, x2 = x, x + w
                y1, y2 = y, y + h
                proposed_bbox = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                
                # check the proposal with every elements of the ground truth boxes
                is_object = False   # define flag
                between_negative_rois = False
                for gt_bbox in gt_bboxes:
                    iou = calculate_IoU(gt_bbox, proposed_bbox)
                    # positive if iou >= 0.5
                    if iou >= IoU_threshold['positive']:
                        # add roi
                        positive_rois.append(proposed_bbox)
                        # add label, here x, y is the center of the rectangle
                        proposed_bbox_xywh = ((x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1))
                        gt_bbox_xywh = ((gt_bbox['x1']+gt_bbox['x2'])/2, (gt_bbox['y1']+gt_bbox['y2'])/2, 
                                        (gt_bbox['x2']-gt_bbox['x1']), (gt_bbox['y2']-gt_bbox['y1']))
                        positive_labels.append([gt_bbox['class'], proposed_bbox_xywh, gt_bbox_xywh])
                        is_object = True
                        break
                    else:
                        if iou >= IoU_threshold['negative']:
                            between_negative_rois = True
                        
                # if the proposal is not close to any ground truth box, and at least iou >= 0.1
                if is_object == False and between_negative_rois:
                    # add roi
                    negative_rois.append(proposed_bbox)
                    # add background label
                    proposed_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                    gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                    negative_labels.append([0, proposed_bbox_xywh, gt_bbox_xywh])
                
            # add to train data
            self.train_rois.append({'img': img, 'positive_rois': positive_rois, 'negative_rois': negative_rois})
            self.train_labels.append({'positive_labels': positive_labels, 'negative_labels': negative_labels})
            print(f"positive : negative number = {len(positive_rois)} : {len(negative_rois)}")

        
        print("[*] Dataset generated! Saving labels to ", self.data_path)
        with open(self.data_path / "train_rois.pkl", "wb") as f:
            pickle.dump(self.train_rois, f)
        with open(self.data_path / "train_labels.pkl", "wb") as f:
            pickle.dump(self.train_labels, f)

# %%
def Fast_RCNN_DataLoader(voc_dataset, cfg, train_cfg, shuffle=True):
    rcnn_dataset = Fast_RCNN_Dataset(voc_dataset, cfg)
    return torch.utils.data.DataLoader(rcnn_dataset, batch_size=train_cfg['batch_size'], shuffle=shuffle)

# %%
fast_rcnn_loader = Fast_RCNN_DataLoader(voc_dataset, config, train_config)


'''
# %% [markdown]
# Model

# %%
class RoI_Pool(nn.Module):
    def __init__(self, output_size=(7, 7)):
        self.output_size = output_size
    
    def forward(self, features, boxes):
        assert features.dim() == 4, "Expected CNN features input of (N, C, H, W)"
        return roi_pool(features, boxes, output_size=self.output_size)

# %%
class VGG_RoI(nn.Module):
    def __init__(self, num_classes):
        """
        num_classes doesn not include background
        """
        super().__init__()

        # VGG16 convolution layer setting, remove the last max-pooling 'M'
        feature_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        
        self.features = models.vgg.make_layers(feature_list)
        self.roi_pool = RoI_Pool((7, 7))
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.classifier = nn.Linear(4096, num_classes + 1)
        self.bbox_reg = nn.Linear(4096, num_classes * 4)
        
        self._initialize_weights()


    def _initialize_weights(self):
        print("[*] Initializing VGG_RoI network...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
       
                
    def forward(self, x):
        x = self.features(x)
        x = self.roi_pool(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        object_classification = self.classfier(x)
        bbox_regression = self.bbox_reg(x)
        return object_classification, bbox_regression
    

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
# Loss function

# %%
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def smooth_l1(self, x):
        if torch.abs(x) < 1:
            return 0.5 * torch.pow(x, 2)
        else:
            return torch.abs(x) - 0.5
        
    def forward(self, preds, targets):
        """
        Calculate the regression loss for bounding box refinement.
        Both shape of preds and targets are [N, 4], where N is the number of RoI.
        """
        res = self.smooth_l1(preds - targets)
        return torch.sum(res)


class MultiTaskLoss(nn.Module):
    def __init__(self, lam=1):
        super().__init__()
        self.lam = lam
        # L_cls: cross entropy loss for classification
        self.cls = nn.CrossEntropyLoss()
        # L_loc: smooth L1 loss for box location regression
        self.loc = SmoothL1Loss()
        
    def indicator(self, category):
        return category != 0
        
    def forward(self, scores, preds, targets):
        """
        Multitask loss, where N is number of RoI
        :param scores: [N, C], C is the class number
        :param preds: [N, 4], 4 is for x, y, w, h for boxes
        :param targets: [N], 0 is the background
        """
        N = targets.shape[0]
        cls_loss = self.cls(scores, targets)
        reg_loss = self.loc(scores[:, self.indicator(targets)],
                            preds[:, self.indicator(preds)])

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
			pbar = tqdm(self.category_classification_loader)
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


'''
