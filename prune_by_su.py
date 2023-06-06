from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Load a model
yolo = YOLO("runs/detect/constrained/weights/last.pt")  # build a new model from scratch
model = yolo.model

ws = []
bs = []

for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        w = m.weight.abs().detach()
        b = m.bias.abs().detach()
        ws.append(w)
        bs.append(b)
        print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())
# keep
factor = 0.5
ws = torch.cat(ws)
threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
print(threshold)


def prune_conv(conv1: Conv, conv2: Conv):
    gamma = conv1.bn.weight.data.detach()
    beta  = conv1.bn.bias.data.detach()
    
    keep_idxs = []
    local_threshold = threshold
    while len(keep_idxs) < 8:
        keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
        local_threshold = local_threshold * 0.5
    n = len(keep_idxs)
    # n = max(int(len(idxs) * 0.8), p)
    print(n / len(gamma) * 100)
    # scale = len(idxs) / n
    conv1.bn.weight.data = gamma[keep_idxs]
    conv1.bn.bias.data   = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
    conv1.conv.out_channels = n
    
    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    if not isinstance(conv2, list):
        conv2 = [conv2]
        
    for item in conv2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:, keep_idxs]
    
def prune(m1, m2):
    if isinstance(m1, C2f):      # C2f as a top conv
        m1 = m1.cv2
    
    if not isinstance(m2, list): # m2 is just one module
        m2 = [m2]
        
    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1
    
    prune_conv(m1, m2)

for name, m in model.named_modules():
    if isinstance(m, Bottleneck):
        prune_conv(m.cv1, m.cv2)
        
seq = model.model
for i in range(3, 9):
    if i in [6, 4, 9]: continue
    prune(seq[i], seq[i+1])
    
detect:Detect = seq[-1]
last_inputs   = [seq[15], seq[18], seq[21]]
colasts       = [seq[16], seq[19], None]
for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
    prune(last_input, [colast, cv2[0], cv3[0]])
    prune(cv2[0], cv2[1])
    prune(cv2[1], cv2[2])
    prune(cv3[0], cv3[1])
    prune(cv3[1], cv3[2])

# ***step4，一定要设置所有参数为需要训练。因为加载后的model他会给弄成false。导致报错
# pipeline：
# 1. 为模型的BN增加L1约束，lambda用1e-2左右
# 2. 剪枝模型，比如用全局阈值
# 3. finetune，一定要注意，此时需要去掉L1约束。最终final的版本一定是去掉的
for name, p in yolo.model.named_parameters():
    p.requires_grad = True
    
# 1. 不能剪枝的layer，其实可以不用约束
# 2. 对于低于全局阈值的，可以删掉整个module
# 3. keep channels，对于保留的channels，他应该能整除n才是最合适的，否则硬件加速比较差
#    n怎么选，一般fp16时，n为8
#                int8时，n为16
#    cp.async.cg.shared
#

# yolo.val()
# yolo.export(format="onnx")
yolo.train(data="ultralytics/datasets/ball.yaml", epochs=100)
print("done")
