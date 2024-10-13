from ultralytics import YOLO
import numpy as np
from app.utils import configure_logger

logger = configure_logger("detectorLogger")
class Detector:
    """
    Simple class to run handle the object detectionpart
    It can be customized to use any model and format the outputs
    """
    
    def __init__(self, model_path="./ai/detection-weights/yolo11s.pt") -> None:
        # model path can be the path to an existing model
        # or the name of an ultralytics model which 
        # will be then downloaded to the current directory
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.exception(f"Error loading model at path: {model_path}.\n {e}",exc_info=True)
            logger.info(f"Loading model at path: './detection-weights/yolo11s.pt'.\n {e}")
            self.model = YOLO("./ai/detection-weights/yolo11s.pt")
    
    
    def detect_object(self, frame):
        assert isinstance(frame, np.ndarray), f'Expected the frame to be of type numpy array got : {type(frame)}'
        
        # Run inference on image
        outputs = self.model(frame)[0]
        
        # Format the outputs to (classname, x_center, y_center, width, height)
        class_names = outputs.names
        detections = list(map(lambda box:[class_names[int(box.cls.cpu().numpy()[0])],
                                     *box.xywh.cpu().numpy()[0]], outputs.boxes))
        
        # assert len(detections[0]) == 5, f"Expected the predictions to have length 5. \
        #                                 Got: len({detections[0]}) == {len(detections[0])}"
        return detections
