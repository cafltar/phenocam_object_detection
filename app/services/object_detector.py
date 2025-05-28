import datetime
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class ObjectDetector:
    def __init__(self, score_threshold=0.3):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()
        self.score_threshold = score_threshold
        self.groups = {
            "vehicles": ["car", "truck", "motorcycle", "bicycle", "bus", "train", "boat", "bike"],
            "animals": ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "person": ["person"],
        }

    def preprocess(self, image):
        preprocess = self.weights.transforms()
        return preprocess(image).unsqueeze(0)

    def postprocess(self, output):
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        detected_objects = []
        other_objects = []
        current_time = datetime.datetime.now().isoformat()
        for box, label, score in zip(boxes, labels, scores):
            if score >= self.score_threshold:
                label_name = self.weights.meta["categories"][label.item()]
                is_hit = False
                for group_name, group_items in self.groups.items():
                    if label_name in group_items:
                        detected_objects.append({
                            "box": box.tolist(),
                            "timestamp": current_time,
                            "label": label_name,
                            "score": score.item()
                        })
                        is_hit = True
                if not is_hit:
                    other_objects.append({
                        "box": box.tolist(),
                        "timestamp": current_time,
                        "label": label_name,
                        "score": score.item()
                    })
        return detected_objects, other_objects

    def detect(self, image):
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return self.postprocess(outputs)

    def detect_batch(self, images):
        """Run detection on a batch of images."""
        input_tensors = [self.preprocess(img) for img in images]
        input_batch = torch.cat(input_tensors, dim=0)
        with torch.no_grad():
            outputs = self.model(input_batch)
        return [self.postprocess([output]) for output in outputs]
