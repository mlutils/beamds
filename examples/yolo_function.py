
from PIL import Image
import requests
from beam import logger

try:
    from examples.yolo_model import YOLOConfig, YOLOBeam
except:
    from yolo_model import YOLOConfig, YOLOBeam

def build_yolo():
    config = YOLOConfig()
    yolo = YOLOBeam(config)
    return yolo

