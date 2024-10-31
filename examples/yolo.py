
from PIL import Image
import requests
from beam import logger

try:
    from examples.yolo_model import YOLOConfig, YOLOBeam
except:
    from yolo_model import YOLOConfig, YOLOBeam

def main():

    logger.info("Running YOLO example...")
    config = YOLOConfig()
    logger.info(f"Config: {config}")
    yolo = YOLOBeam(config)
    logger.info(f"Processor: {yolo}")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    logger.info(f"Image: {image}")
    image = yolo.process(image)
    logger.info(f"Processed image: {image}")

    from IPython.display import display
    display(image)
    # store to bundle
    logger.info(f"Storing the processor to bundle: {config.path_to_bundle}")
    yolo.to_bundle(config.path_to_bundle, blacklist=['torch', 'torchvision'])
    print('Done!')


if __name__ == '__main__':
    main()