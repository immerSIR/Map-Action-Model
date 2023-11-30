from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


model.train(data="data.yaml", epochs=20)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/yugo19/Documents/Exp-data/yolo-data/images/dumpgarbage103_jpeg.rf.fe62527242b552a1d2ad0f12f83dacd4.jpg")  # predict on an image
path = model.export(format="onnx")