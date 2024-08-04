from typing import Union, List, Set

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw
from beam import Processor, BeamConfig, BeamParam


class YOLOConfig(BeamConfig):
    parameters = [
        BeamParam('model', str, 'facebook/detr-resnet-50', 'The model to use for object detection.'),
        BeamParam('path-to-bundle', str, '/tmp/yolo-bundle',)
    ]


class YOLOBeam(Processor):

    def __init__(self, config: YOLOConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Load the processor and model
        self.processor = None
        self.model = None
        self.load_hf_model()

    def load_hf_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.get_hparam('model'))
        self.model = AutoModelForObjectDetection.from_pretrained(self.get_hparam('model'))

    @classmethod
    @property
    def excluded_attributes(cls) -> set[str]:
        return super().excluded_attributes | {"processor", "model"}

    def load_state_dict(self, path, ext=None, exclude: Union[List, Set] = None, hparams=True, exclude_hparams=None,
                        overwrite_hparams=None, **kwargs):
        super().load_state_dict(path, ext, exclude, hparams, exclude_hparams, overwrite_hparams, **kwargs)
        self.load_hf_model()

    def process(self, image: Image):
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        # Draw the results
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{self.model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")
        return image

