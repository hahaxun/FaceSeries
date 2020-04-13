'''
add some interface for reconstruct network
'''
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, backbone_name:str, device, INPUT_SIZE, BACKBONE_RESUME_ROOT):
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
    def __init__(self, head_name, device, NUM_CLASS, GPU_ID, EMBEDDING_SIZE, HEAD_RESUME_ROOT):
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

        
    



