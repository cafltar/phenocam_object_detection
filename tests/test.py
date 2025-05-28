import sys
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import requests
from io import BytesIO
import json
import datetime
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Define model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()  # Set the model to evaluation mode

groups = {
    "vehicles": ["car", "truck", "motorcycle", "bicycle", "bus", "train", "boat", "bike"],
    "animals": ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "person": ["person"],
}

# Function to preprocess image
def preprocess(image):
    preprocess = weights.transforms()
    return preprocess(image).unsqueeze(0)

# Function to postprocess results
def postprocess(output, threshold=0.3):
    boxes = output[0]['boxes']
    labels = output[0]['labels']
    scores = output[0]['scores']
    
    detected_objects = []
    other_objects = []

    current_time = datetime.datetime.now().isoformat()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            label_name = weights.meta["categories"][label.item()]
            is_hit = False
            for group_name, group_items in groups.items():
                
                if label_name in group_items:  # Check if the label belongs to any group
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

# Main function for testing with a local jpg file
def main():
    # Load the image from a local file
    image_path = "test.jpg"
    img = Image.open(image_path).convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(img)

    # Run the model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Postprocess the outputs
    detections, others = postprocess(outputs)

    # Convert detections to JSON
    detections_json = json.dumps(detections, indent=4)
    others_json = json.dumps(others, indent=4)

    # Print the JSON output
    print(detections_json)
    print(others_json)

if __name__ == "__main__":
    main()