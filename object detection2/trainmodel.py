import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="EV3brickset")
trainer.setTrainConfig(object_names_array=["EV3brick"], batch_size=8, num_experiments=50, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()
