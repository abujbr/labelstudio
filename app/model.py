import os
import random
import requests
from PIL import Image
from io import BytesIO
from yolov5.models.experimental import attempt_load
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
import logging
import torch


logger = logging.getLogger(__name__)

# URL with host
LS_URL =  "http://localhost:8080"
#LS_URL = "http://192.168.100.3:8080"
LS_API_TOKEN = "cde3fa5b0dd48a07335e5adc06859a228afe1f50"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['Socket-gray', 'Socket-green', 'rack']
        # Load model
        #self.model = attempt_load('best2.pt')
        self.model = torch.hub.load("ultralytics/yolov5:v7.0", "custom", path='best2.pt', trust_repo=True)

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        
        # Getting URL and loading image
        #image = Image.open(BytesIO(requests.get(full_url, headers=header).content))
        try:
           image = Image.open(BytesIO(requests.get(full_url, headers=header).content))
        except Exception as e:
           print(f"Error loading image: {str(e)}")
          
        # Height and width of image
        img_w, img_h  = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        results = []
        all_scores = []

        # Getting prediction using model
        objs = self.model(image)
        
        lowest_conf = 2.0

        img_results = []

        for obj in objs.pred[0]:
            x, y, w, h, conf, cls = obj[:6]
            cls = int(cls)
            conf = float(conf)
            print(x)
            print(y)
            print(w)
            print(h)
            print(img_w)
            print(img_h)

            x_normalized = 100 * float(x) / img_w
            y_normalized = 100 * float(y) / img_h
            w_normalized = 100 * float(w) / img_w
            h_normalized = 100 * float(h) / img_h
            w_normalized = w_normalized - x_normalized
            h_normalized = h_normalized - y_normalized
          
            print(x_normalized)
            print(y_normalized)
            print(w_normalized)
            print(h_normalized)
            print(conf)
            if conf < lowest_conf:
                lowest_conf = conf
            label = self.labels[cls]
            if (conf > 0.80):
              img_results.append({
                  'from_name': self.from_name,
                  'to_name': self.to_name,
                  'type': 'rectanglelabels',
                  'value': {
                      'rectanglelabels': [label],
                      'x': x_normalized,
                      'y': y_normalized,
                      'width': w_normalized,
                      'height': h_normalized,
                  },
                  'score': conf
              })

        # Calculating score
        #score += conf.item()
        result = {
            'result': img_results
        }
        return [result]
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}