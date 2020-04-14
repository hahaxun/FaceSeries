'''
add some interface for reconstruct network
'''
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from config import cfg
from detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import torch.nn as nn
import torch

class Detection(nn.Module):
    def __init__(self. device):
        self.device = device
        self.detector = detect_faces
    
    def __call__(self, data):
        return self.detector(data)

class Embedding(nn.Module):
    def __init__(self, device, backbone_name = cfg['BACKBONE_NAME'], INPUT_SIZE = cfg['INPUT_SIZE'], BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']):
        BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                    'ResNet_101': ResNet_101(INPUT_SIZE), 
                    'ResNet_152': ResNet_152(INPUT_SIZE),
                    'IR_50': IR_50(INPUT_SIZE), 
                    'IR_101': IR_101(INPUT_SIZE), 
                    'IR_152': IR_152(INPUT_SIZE),
                    'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                    'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                    'IR_SE_152': IR_SE_152(INPUT_SIZE)}

        self.device = device
        self.embedding = BACKBONE_DICT[backbone_name].load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        self.embedding = self.embedding.to(device)
    
    def __call__(self, data):
        data = data.to(self.device)
        return self.embedding.forward()


class MetricHead(nn.Module):
    def __init__(self, device, head_name = cfg['HEAD_NAME'], NUM_CLASS = cfg['NUM_CLASS'], GPU_ID = cfg['GPU_ID'], EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'], HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']):
        HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
        self.device = device
        self.head = HEAD_DICT[head_name].load_state_dict(torch.load(HEAD_RESUME_ROOT))
        self.head = self.head.to(device)
    
    def __call__(self, features_left, features_right):
        features_left = features_left.to(self.device)
        features_right = features_right.to(self.device)
        return self.head(features_left,features_right)

        
    
class FaceEncoer(nn.Module):
    def __init__(self, device, crop_size = 112):
        #default format is N X H X W X D
        super().__init__("FaceEncoder")
        self.detection = Detection(device)
        self.align = align
        self.embedding = Embedding(device)
        self.crop_size = crop_size
        self.scale = crop_size / 112.
        self.reference = get_reference_facial_points(default_square = True) * scale

    def warpface(self, landmarks, img):
        faces = []
        for landmark  in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            faces.append(warp_and_crop_face(np.array(img), facial5points, self.reference, crop_size=(self.crop_size, self.crop_size)))
        return faces


    def build(self, image):
        #1.add face detection
        bounding_boxes, landmarks = detect_faces(image)
        #2.add face align
        wrapedfaces = warpface(landmarks,image)
        #3.compute face embedding
        return self.embedding(wrapedfaces)


