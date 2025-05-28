import os
import sys
import json
import logging
from services.file_loader import FileLoader
from services.object_detector import ObjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main(url, score_threshold=None):
    config = load_config()
    if score_threshold is None:
        score_threshold = config.get('score_threshold', 0.5)
    try:
        logger.info(f'Loading: {url}')
        loader = FileLoader(url)
        img = loader.load_image()
        logger.info('Preprocessing and running detection...')
        detector = ObjectDetector(score_threshold)
        detections, others = detector.detect(img)
        logger.info('Converting to json...')
        detections_json = json.dumps(detections, indent=4)
        others_json = json.dumps(others, indent=4)
        print(detections_json)
        print(others_json)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    #time.sleep(900)  # 900 seconds = 15 minutes

if __name__ == "__main__":
    logger.info('Running v0.1.2')
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py image_url [score_threshold]")
        sys.exit(1)
    image_url = sys.argv[1]
    score_threshold = None
    if len(sys.argv) == 3:
        score_threshold = float(sys.argv[2])
    main(image_url, score_threshold)