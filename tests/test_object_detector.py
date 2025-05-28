import sys
import os
from PIL import Image
import pytest
# To run this test, use the following command from the CAFLTAR_object_detection directory:
# python -m pytest tests/test_object_detector.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from services.object_detector import ObjectDetector

@pytest.mark.parametrize("img_filename", ["two-person.jpg"])
def test_object_detector_person(img_filename):
    # Load test image
    img_path = os.path.join(os.path.dirname(__file__), 'assets', img_filename)
    img = Image.open(img_path).convert('RGB')
    # Create detector
    detector = ObjectDetector(score_threshold=0.5)
    # Run detection
    detections, others = detector.detect(img)
    print('Detections:')
    print(detections)
    print('Other objects:')
    print(others)
    # Optionally, add assertions here if you want to check for expected results
    assert isinstance(detections, list)
    assert len(detections) > 0
    assert detections[0]['label'] == 'person'

@pytest.mark.parametrize("img_filename", ["blank-field.jpg"])
def test_object_detector_blank_field(img_filename):
    img_path = os.path.join(os.path.dirname(__file__), 'assets', img_filename)
    img = Image.open(img_path).convert('RGB')
    detector = ObjectDetector(score_threshold=0.5)
    detections, others = detector.detect(img)
    print('Detections:')
    print(detections)
    print('Other objects:')
    print(others)
    assert isinstance(detections, list)
    assert len(detections) == 0
