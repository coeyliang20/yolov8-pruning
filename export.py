from ultralytics import YOLO
import sys

# Load a model
model = YOLO(sys.argv[1])  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="VOC.yaml", epochs=100)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
print(success)