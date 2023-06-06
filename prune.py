from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck,Conv,C2f,SPPF,Detect
from torch import nn

def prune_conv(conv1:Conv,conv2:Conv):
    # thinking the sentivity of bn, min(gama) whether nearly equal to zer0
    # torch.sort(gamma,descending=True)[0].min()
    gamma = conv1.bn.weight.data.detach()
    beta  = conv1.bn.bias.data.detach()
    # 
    keep_idx = []
    local_threshold = threshold
    
    while len(keep_idx) < 8:
        keep_idx = torch.where(gamma.abs()>=local_threshold)[0]
        local_threshold = local_threshold * 0.5
    n = len(keep_idx)
    # how much channels we keep
    print(n/len(gamma)*100)
    # prune bn
    conv1.bn.weight.data = gamma[keep_idx]
    conv1.bn.bias.data   = beta[keep_idx]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idx]
    conv1.bn.running_mean = conv1.bn.running_mean.data[keep_idx]
    conv1.bn.num_features = n
    # prune conv channel
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idx]
    conv1.conv.out_channels = n
    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idx]
    
    # prune conv2
    if not isinstance(conv2,list):
        conv2 = [conv2]
    
    for item in conv2:
        if item is not None:
            if isinstance(item,Conv):
                conv = item.conv
            else:
                # Conv2d 
                conv = item
            
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:,keep_idx]       
            
def prune(m1,m2):
    if isinstance(m1,C2f):
        m1 = m1.cv2
    if not isinstance(m2,list):
        m2 = [m2]
    for i,item in enumerate(m2):
        if isinstance(item,C2f) or isinstance(item,SPPF):
            m2[i] = item.cv1
    prune_conv(m1,m2)       
                

if __name__ == "__main__":
    yolo = YOLO("/mnt/jansen/yolov8/ultralytics-main/runs/detect/constrained/weights/last.pt")
    model = yolo.model

    ws = []
    bs = []

    for k,m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            # m.weight meas gamma
            w = m.weight.abs().detach() 
            b = m.bias.abs().detach()
            ws.append(w)        
            bs.append(b)
            # print(k,w.max().item(),w.min().item(),b.max().item(),b.min().item())
            print(w.min().item())

    factor = 0.99
    
    ws = torch.cat(ws)
    threshold = torch.sort(ws,descending=True)[0][int(len(ws)*factor)]
    print(threshold)    

    # global pruning for Bottleneck
    for name,m in model.named_modules():
        if isinstance(m,Bottleneck):
            prune_conv(m.cv1,m.cv2)
    seq = model.model
    
    for i in range(3,9):
        if i in [4,6,9]: continue
        prune(seq[i],seq[i+1])
    
    detect:Detect = seq[-1]
    last_inputs = [seq[15],seq[18],seq[21]]
    colasts = [seq[16],seq[19],None]
    
    for last_input,colast,cv2,cv3 in zip(last_inputs,colasts,detect.cv2,detect.cv3):
        prune(last_input,[colast,cv2[0],cv3[0]])
        prune(cv2[0], cv2[1])
        prune(cv2[1], cv2[2])
        prune(cv3[0], cv3[1])
        prune(cv3[1], cv3[2])
        
    
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True
        
    yolo.val()
    # yolo.export(format='onnx')