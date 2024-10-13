from time import sleep
import numpy as np
from ai.tracker.kalman_filter import KalmanFilter
import cv2
from scipy.spatial.distance import mahalanobis, euclidean
from app.utils import configure_logger


logger = configure_logger('kalmanTrackerLogger')

class TrackedObject:
    """
    Represents a tracked object using a Kalman Filter.

    Attributes:
        id (int): Unique identifier for the object.
        state (numpy.ndarray): Current state of the object.
        age (int): How long the object has been tracked.
        lost_frames (int): Count of frames where the object was not detected.
        max_lost_frames (int): Maximum frames to wait before deleting the object.
        __kalman__ (KalmanFilter): The Kalman filter associated with the object.
        acceleration (bool): Whether to consider acceleration for estimation or not
    """
    def __init__(self, id, state=None, age=0, acceleration:bool = False) :
        """
        Initializes a tracked object and its Kalman filter.

        Args:
            id (int): Unique identifier for the object.
            state (numpy.ndarray, optional): Initial state of the object. Defaults to None.
            age (int, optional): The age of the object (in frames). Defaults to 0.

        Raises:
            AssertionError: If `id` or `age` are not integers.
        """
        
        assert isinstance(id,int), f'Type(id) = {type(id)} != int'
        assert isinstance(age,int), f'Type(age) = {type(age)} != int'
        
        self.acceleration = acceleration # Whether to consider acceleration during estimation or not
        
        # initialize a kalman filter with 4 dynamic variables 
        # (x,vx,y,vy)
        # and 2 measurements
        # (x,y)
        self.initialize() 
        self.id = id
        
        if state != None:
            self.state = state
        self.age = age           # How long the object has been tracked
        self.lost_frames = 0     # Count of frames since last detection
        self.max_lost_frames = 5  # Tolerance before deleting the object
        
        
    @property
    def state(self):
        return self.__kalman__.post_state    
    
    @state.setter
    def state(self, value):
        """Sets the current state of the object."""
        
        # To make sure the state is not assigned any value
        assert isinstance(value, np.ndarray), f'Expected the state to be of type numpy array got {type(value)}'
        assert value.shape == self.__kalman__.post_state.shape, f'New value shape : {value.shape} != {self.__kalman__.post_state.shape} Expected shape'
        self.__kalman__.post_state  = value
    
    @property
    def pred_state(self):
        """Returns the predicted state of the object from the Kalman filter."""
        return self.__kalman__.predicted_state
    
    @property
    def cov(self):
        """Returns the residual covariance of the object from the Kalman filter."""
        return self.__kalman__.residual_cov    
    
    
    def predict(self):
        """
        Uses the Kalman filter to predict the next state of the object.

        Returns:
            np.ndarray: The predicted state.
        """
        # Use Kalman filter to predict next state
        self.__kalman__.predict_step()
        return self.pred_state

    def update(self, measurements):
        """
        Updates the Kalman filter with new measurements.

        Args:
            measurements (np.ndarray): New measurements (e.g., bounding box center coordinates).
        """
        # Update with new detection (bounding box)
        self.__kalman__.correct_step(measurements)
        self.lost_frames = 0  # Reset lost frame counter when updated

    def mark_lost(self):
        # Increase lost frame count when no detection is found
        self.lost_frames += 1

    def is_deleted(self):
        # Check if the object should be deleted
        return self.lost_frames > self.max_lost_frames
    
    def track(self,box, frame):
        # Runs one iteration of the Kalman Filter tracking algorithm:
        # update -> predict
        class_, x_c, y_c, w, h = box  # Center point (x, y) and width/height (w, h)
            
        # Measurement update (use the center of the bounding box as the measurement)
        measurements = np.array([[np.float32(x_c)], 
                                [np.float32(y_c)]])
        
        # Update the state of the Kalman filter
        self.update(measurements)
        # Predict the next position of the object
        self.predict()

        # Draw the boxes + center point for tracking
        x = int(x_c - w/2)
        y = int(y_c - h/2)
        x2 = int(x_c + w/2)
        y2 = int(y_c + h/2)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_} {self.id +1}',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Circle in green to indicate that the object is detected + tracked
        predicted_x, predicted_y = int(self.pred_state[0])%frame.shape[1], \
                                    int(self.pred_state[2])%frame.shape[0]
        
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 255, 0), -1)
        
    
    def initialize(self):
        """
        Initializes the Kalman filter for the object based on the dynamic model.

        Args:
            acceleration (bool, optional): If True, includes acceleration in the state variables.
                                           Defaults to False.
        """
        if not self.acceleration:
            # We assume that we want to track position and velocity of the object in 2D
            # So we need 4 dynamic variables(position_x, velocity_x, position_y,velocity_y)
            # We can get the position of the object. So the number of measurement is 2 (position_x, position_y)
            self.__kalman__ = KalmanFilter(nb_dynamics=4, nb_measurements=2)
            # Only the positions of the object are available for measurement
            # Thus only the position variables will be set to 1
            
            self.__kalman__.measurement_matrix = np.array([[1, 0, 0, 0], # (x,0,0,0)
                                            [0, 0, 1, 0]], np.float32) # (0,0,y,0)
            # Motion equation in 2D (assuming constant velocity) is
            # x(t) = x(t-1) + v_x(t-1)*delta_t 
            # v_x(t) = v_x(t-1)
            # y(t) = x(t-1) + v_x(t-1)*delta_t 
            # v_y(t) = v_y(t-1)
            self.__kalman__.transition_matrix = np.array([[1, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 1]], np.float32)
            
            # Set random process & measurement noise
            # These parameters can be tuned based on trial in real scenarios
            self.__kalman__.process_noise_cov = np.eye(4, dtype=np.float32) * 1e-3
            self.__kalman__.measurement_noise_cov = np.eye(2, dtype=np.float32) * 1e-2 
        else:
            # We assume that we want to track position and velocity of the object in 2D. But since the acceleration is not null
            # So we need 6 dynamic variables(position_x, velocity_x, position_y,velocity_y, acc_x, acc_y)
            # We can get the position of the object. So the number of measurement is 2 (position_x, position_y)
            self.__kalman__ = KalmanFilter(nb_dynamics=6, nb_measurements=2)
            # Only the positions of the object are available for measurement
            # Thus only the position variables will be set to 1
            
            self.__kalman__.measurement_matrix = np.array([[1, 0, 0, 0,0,0], # (x,0,0,0)
                                            [0, 0, 1, 0,0,0]], np.float32) # (0,0,y,0)
            # Motion equation in 2D (assuming constant acceleration) is
            # x(t) = x(t-1) + v_x(t-1)*delta_t +1/2 acc_x(t-1)*delta_t**2
            # v_x(t) = v_x(t-1)+ acc_x(t-1)*delta_t
            # y(t) = y(t-1) + v_y(t-1)*delta_t +1/2 acc_y*delta_t**2
            # v_y(t) = v_y(t-1)+ acc_y(t-1)*delta_t
            # acc_x(t) = acc_x(t-1)
            # acc_y(t) = acc_y(t-1)
            self.__kalman__.transition_matrix = np.array([[1, 1, 0, 0,1/2,0],
                                        [0, 1, 0, 0,1,0],
                                        [0, 0, 1, 1,0,1/2],
                                        [0, 0, 0, 1,0,1],
                                        [0, 0, 0, 0,1,0],
                                        [0, 0, 0, 0,0,1]], np.float32)
            
            # Set random process & measurement noise
            # These parameters can be tuned based on trial in real scenarios
            self.__kalman__.process_noise_cov = np.eye(6, dtype=np.float32) * 1e-3
            self.__kalman__.measurement_noise_cov = np.eye(2, dtype=np.float32) * 1e-2 

class KalmanTracker:
    """
    Tracks multiple objects using Kalman filters. Can operate in single or multi-object tracking mode.

    Attributes:
        mode (str): Tracking mode ('single' or 'multi').
        tracked_objects (list[TrackedObject]): List of currently tracked objects.
    """
    
    def __init__(self,mode: str ="single", estimate_acceleration: bool= False, association_metric: str="euclidean") -> None:
        """
        Initializes the Kalman tracker.

        Args:
            mode (str, optional): Tracking mode. Defaults to "single".

        Raises:
            AssertionError: If mode is not 'single' or 'multi'.
        """
        
        assert (mode == 'single') | (mode == 'multi'), f'Unknown mode : {mode}'
        assert (association_metric == 'euclidean') | (association_metric == 'mahalanobis'), f'Association Metric : {association_metric} not supported!'
        self.mode = mode
        self.tracked_objects = []
        self.number_objects = 0
        self.estimate_acceleration = estimate_acceleration
        self.association_metric = association_metric
        
    def __search_best_fit__(self,
                            candidates: list,
                            tracked_object: TrackedObject,
                            distance: str ="mahalanobis", 
                            max_threshold_distance: int =50,
                            ) -> int|None:
        
        """
        Searches for the best candidate that matches the tracked object.

        Args:
            candidates (list): List of candidate bounding box centers.
            tracked_object (TrackedObject): The tracked object.
            distance (str, optional): Distance metric ('euclidean' or 'mahalanobis'). Defaults to "mahalanobis".
            max_threshold_distance (int, optional): Maximum allowable distance for association. Defaults to 12.

        Returns:
            int | None: The index of the best candidate if found, None otherwise.
        """
        
        if len(candidates) == 0:
            return None
        
        position_pred = int(tracked_object.pred_state[0]), int(tracked_object.pred_state[2]) 
        
        if distance == "euclidean":
            distances = [euclidean(np.array(position_pred,np.int16), np.array(box_center,np.int16)) for box_center in candidates]
        
        elif distance == "mahalanobis":
            # residual covariance could be used in this case
            try:
                QI = np.linalg.inv(tracked_object.cov)
            except Exception as e:
                logger.exception(f"Exception when computing the inverse of the Matrix QI.\n {e}", exc_info=True)
                raise e
            
            distances = [mahalanobis(np.array(position_pred),
                                     np.array(box_center),
                                     QI) for box_center in candidates]
        else:
            raise NotImplementedError("Other distances than euclidean and Mahalanobis are not supported yet!")
       
        min_idx = int(np.argmin(distances))
        # To make sure the tracker isn't associated with 
        # wrong objects
        return min_idx if distances[min_idx] <= max_threshold_distance else None
    
    
    def initialize(self, boxes, frame):
        
        assert len(boxes) >0, f"Boxes array is empty"
        assert sum([len(box)== 5 for box in boxes]) == len(boxes), f"Not all boxes have the following format:\
                                                                (classname, x_c,y_c,width, height).\nGot: {boxes}"
        assert isinstance(frame, np.ndarray), f'Expected the frame to be of type numpy array got : {type(frame)}'
        
        initial_estimates= {}
        if self.mode == "single":
            
            initial_estimates = initial_estimates | self.__add_new_tracked_object__(boxes[0], frame)
        
        else:
            # Track all detected objects
            for idx, box in enumerate(boxes):
                
                initial_estimates = initial_estimates | self.__add_new_tracked_object__(box, frame)
            
            logger.info(f'Total of: {idx} objects are being tracked')
        return initial_estimates        
                
        
    def __add_new_tracked_object__(self, box: list, frame: np.ndarray) -> None:
        """
        Adds new TrackedObjects to the list of tracked_objects.
        Args:
            box (list): Detected bounding box.
            frame (np.ndarray): The current video frame.
            acceleration (bool): Whether to include acceleration in the state model.
        """
        initial_estimates = {}
        obj = TrackedObject(id=self.number_objects,acceleration=self.estimate_acceleration)
        obj.track(box, frame)     
        self.tracked_objects.append(obj)
        
        # Update the dictionary of id_estimates
        initial_estimates[obj.id] = ([float(obj.state[1]), #velocity
                                float(obj.state[3])],
                                [int(obj.state[0]), #displacement
                                int(obj.state[2])])
        self.number_objects += 1
        return initial_estimates
            

    def clean_deleted(self) -> None:
        """Removes objects that are marked for deletion."""
        self.tracked_objects = [obj for obj in self.tracked_objects if not obj.is_deleted()]

    def update(self,boxes, frame):
        
        
        assert isinstance(frame, np.ndarray), f'Expected the frame to be of type numpy array got : {type(frame)}'
        
        # dictionary of estimated variables: velocity+position
        id_estimates = {}
        
        if len(boxes) > 0:
            
            if len(self.tracked_objects) == 0 :
                # Initialize the trackers
                id_estimates = self.initialize(boxes, frame)
            
            else:
                candidates = list(map(lambda box: (box[1], box[2]), boxes))
                for obj in self.tracked_objects:
                    # Associate past trackers with detections
                    
                    best_id = self.__search_best_fit__(candidates,obj,self.association_metric)
                    
                    if best_id == None:
                        # Couldn't find a matching object => object lost
                        obj.mark_lost()
                        
                        
                        # Perform a prediction step
                        predicted_state = obj.predict()
                        logger.info(f'Object with id : {obj.id} is lost. Estimated state (x,vx,y,vy) is : {predicted_state}')
                        
                        # Draw an orange dot to indicate that the object is lost
                        # state = (x,vx,y,vy)
                        predicted_x, predicted_y = int(predicted_state[0])%frame.shape[1], int(predicted_state[2])%frame.shape[0]

                        # Draw the predicted position
                        # Orange: to indicate that the object is lost
                        # Red: to indicate that the object will be deleted
                        # if it doesn't appear again in the next frame
                        orange = (0, 128, 255)
                        red = (0, 0, 255)
                        almost_reached_threshold = obj.lost_frames == obj.max_lost_frames - 1
                        cv2.circle(frame, (predicted_x, predicted_y), 5, red if almost_reached_threshold else orange , -1)
                        
                        # Update the dictionary of id_estimates
                        id_estimates[obj.id] = ([float(predicted_state[1]), float(predicted_state[3])], #velocity
                                            [predicted_x, predicted_y]) # positions
                        
                    else:
                        
                        # Associated the object successfully
                        candidates.pop(best_id)
                        box = boxes.pop(best_id)
                        obj.track(box, frame)
                        # Update the dictionary of id_estimates
                        id_estimates[obj.id] = ([float(obj.state[1]), #velocity
                                                float(obj.state[3])],
                                                [int(obj.state[0]), #displacement
                                                int(obj.state[2])])
                  
        else:
            # Check if there are already tracked objects
            if len(self.tracked_objects) == 0 :
                # Skip
                return frame, None
            
            else:
                # Perform a prediction step
                for obj in self.tracked_objects :
                    
                    # Update the state of the object
                    obj.mark_lost()
                    # Perform a prediction step
                    predicted_state = obj.predict()
                    logger.info(f'Object with id : {obj.id} is lost. Estimated state (x,vx,y,vy) is : {predicted_state}')
                    # Draw an orange dot to indicate that the object is lost
                    # state = (x,vx,y,vy)
                    predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[2])

                    # Draw the predicted position
                    # Orange: to indicate that the object is lost
                    # Red: to indicate that the object will be deleted
                    # if it doesn't appear again in the next frame
                    orange = (0, 128, 255)
                    red = (0, 0, 255)
                    almost_reached_threshold = obj.lost_frames == obj.max_lost_frames - 1
                    cv2.circle(frame, (predicted_x, predicted_y), 5, red if almost_reached_threshold else orange , -1)
                    
                    # Update the dictionary of id_estimates
                    id_estimates[obj.id] = ([float(predicted_state[1]), float(predicted_state[3])], #velocity
                                            [predicted_x, predicted_y]) # positions
                    
        
        # Remove Untracked objects
        self.clean_deleted()
        
        return frame, id_estimates
