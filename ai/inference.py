from time import sleep
import numpy as np
from tracker.kalman_filter import KalmanFilter
import cv2
from ultralytics import YOLO
from scipy.spatial.distance import mahalanobis, euclidean
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


class Tracker:
    
    # define a class variable that increments
    # Whenever the class is instantiated
    id = 0
    
    def __init__(self,tracker: str = "kalman") -> None:
        
        self.id_ = Tracker.id
        Tracker.id += 1
        
        if tracker == "kalman":
            # Use Kalman Filter tracker
            # We assume that we want to track position and velocity of the object in 2D
            # So we need 4 dynamic variables(position_x, velocity_x, position_y,velocity_y)
            # We can get the 
            self.tracker = KalmanFilter(nb_dynamics=4, nb_measurements=2)
            # Only the positions of the object are available for measurement
            # 
            self.tracker.measurement_matrix = np.array([[1, 0, 0, 0], # (x,0,0,0)
                                         [0, 0, 1, 0]], np.float32) # (0,0,y,0)
            # Motion equation in 2D (assuming constant velocity) is
            # x(t) = x(t-1) + v_x(t-1)*delta_t 
            # v_x(t) = v_x(t-1)
            # y(t) = x(t-1) + v_x(t-1)*delta_t 
            # v_y(t) = v_y(t-1)
            self.tracker.transition_matrix = np.array([[1, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 1]], np.float32)
            
            # Set random process & measurement noise
            # These parameters can be tuned based on trial in real scenarios
            self.tracker.process_noise_cov = np.eye(4, dtype=np.float32) * 1e-3
            self.tracker.measurement_noise_cov = np.eye(2, dtype=np.float32) * 1e-2            
            
        else:
            raise NotImplemented("Other Methods are not implemented yet!")
        
    def __search_best_fit__(self,
                            candidates: list,
                            distance: str ="euclidean", 
                            max_threshold_distance: int =10):
        
        # Compute distances between predicted state and each bounding box
        position_pred = int(self.tracker.predicted_state[0]), int(self.tracker.predicted_state[2]) 
        
        if distance == "euclidean":
            distances = [euclidean(np.array(position_pred), np.array(box_center)) for box_center in candidates]
        else:
            raise NotImplementedError("Other distances than euclidean are not supported yet!")
        
        min_distance = min(distances)
        
        if min_distance < max_threshold_distance :
            # To make sure the tracker isn't associated with 
            # wrong objects
            id_min = np.argmin(distances, axis=0)
            return id_min
        
        
        return None
    
    def track(self,object_class,
                 video_path, 
                 model_path="./detection-weights/yolo11s.pt",
                 #tracking_type="single_object",
                  ):
        
        # # initialize the state if an initial position exists
        # if isinstance(initial_position, np.ndarray) and initial_position.size > 0: 
            
        #     self.tracker.predicted_state = initial_position.astype(np.float32)
        #     self.tracker.post_state = initial_position.astype(np.float32)
        
        # Load the model
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        iteration = 0
        speed = []
        distance = []
        
        displacement_x = []
        displacement_y = []
        
        velocity_x = []
        velocity_y = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in the frame using YOLO
            outputs = model(frame)
            
            # Filter predicted boxes
            boxes = list(filter(lambda box:  int(box.cls.cpu().numpy()[0]) == object_class, outputs[0].boxes))
            
            if len(boxes) > 0:
                #boxes = format(boxes)

                if iteration == 0:
                    # Assume tracking a random detected object (for simplicity)
                    
                    x_c, y_c, w, h = (boxes[3].xywh.cpu().numpy()).astype(int)[0]  # Center point (x, y) and width/height (w, h)
                    #TODO remove this print
                    print(f'Confidence: {boxes[3].conf}')
                    
                    # Measurement update (use the center of the bounding box as the measurement)
                    measured_x = x_c
                    measured_y = y_c
                    
                    x = int(x_c) - w//2
                    y = int(y_c) - h//2
                    x2 = int(x_c) + w//2
                    y2 = int(y_c) + h//2
                    
                    self.tracker.correct_step(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
                else:
                    # Search for the closest bounding box to the prediction [[cx,cy]]
                    center_points = list(map(lambda box: np.array([box.xywh.cpu().numpy()[0][0] ,
                                                                   box.xywh.cpu().numpy()[0][1] ]),
                                             boxes))
                    
                    id_min = self.__search_best_fit__(center_points)
                    
                    if id_min != None:
                        measured_x, measured_y = center_points[id_min]
                        self.tracker.correct_step(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
                    
                        x,y,x2,y2 = (boxes[id_min].xyxy.cpu().numpy()).astype(int)[0]    
                    
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Senator {self.id_+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
            else:
                # Predict next state (no measurement available)
                # self.tracker.predict_step()
                pass
                

            # Predict the next position
            predicted, _ = self.tracker.predict_step() # predicted_state = (x,vx,y,vy)
            print(f'Prediction: {predicted}')
            predicted_x, predicted_y = int(predicted[0]), int(predicted[2])

            # Draw the predicted position
            cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

            # Display the resulting frame
            cv2.imshow('Object Tracking', frame)
        
            displacement_x.append(predicted[0])
            displacement_y.append(predicted[2])
            
            velocity_x.append(predicted[1])
            velocity_y.append(predicted[3])
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            iteration += 1

        cap.release()
        cv2.destroyAllWindows()
    
        speed = np.sqrt(np.power(velocity_x,2) + np.power(velocity_y,2))
        distance = np.sqrt(np.power(displacement_x,2) + np.power(displacement_y,2))
        
        fig,axes = plt.subplots(1,1,figsize=(10,8))
        
        axes.plot(range(len(speed)), speed, label="speed")
        axes.plot(range(len(distance)), distance, label="distance")
        
        
        axes.plot(range(len(displacement_x)), displacement_x, label="displacement_x")
        axes.plot(range(len(displacement_y)), displacement_y, label="displacement_y")
        axes.plot(range(len(displacement_x)), velocity_x, label="velocity_x")
        axes.plot(range(len(displacement_y)), velocity_y, label="velocity_y")
        
        axes.set_xlabel(f"Time x27/{iteration} (sec)")
        axes.set_ylabel("Displacement/Velocity (pixel/sec)")
        
        plt.legend()
        plt.show()
        
        


if __name__ == "__main__":
    
    video_path = "../videos/video_senators.mp4"
    tracker = Tracker()
    tracker.track(object_class=0,
                  video_path=video_path,# Person class
                  ) 
    
    
    # # # Load the model
    # model = YOLO("yolo11s.pt")

    # # Export the model to ONNX format
    # model.export(format="onnx")