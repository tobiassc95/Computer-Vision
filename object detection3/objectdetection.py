import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-044--loss-0002.569.h5") 
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections =detector.detectObjectsFromImage(input_image="input/EV3bricktest21.jpg", output_image_path="output/EV3bricktested21.jpg",minimum_percentage_probability=30)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])



# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from imageai.Detection import ObjectDetection

# detector = ObjectDetection()
# model_path = "models/yolo-tiny.h5"
# input_path = "input/inputimg.jpg"
# output_path = "output/outputimg.jpg"
# # model_path = "c:/Users/Tobias Scala/Documents/Visual Studio Code/mecatronica/object detection/models/yolo-tiny.h5"
# # input_path = "c:/Users/Tobias Scala/Documents/Visual Studio Code/mecatronica/object detection/input/inputimg.jpg"
# # output_path = "c:/Users/Tobias Scala/Documents/Visual Studio Code/mecatronica/object detection/output/outputimg.jpg"

# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath(model_path)
# detector.loadModel()
# detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
# for eachItem in detection:
#     print(eachItem["name"] , " : ", eachItem["percentage_probability"])