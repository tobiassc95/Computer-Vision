# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from imageai.Detection import ObjectDetection

detector = ObjectDetection()
model_path = "models/yolo-tiny.h5"
input_path = "input/inputimg2.jpg"
output_path = "output/outputimg22.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])