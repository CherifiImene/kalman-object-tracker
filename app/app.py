from collections import deque
import json
import os
import cv2
import numpy as np
from ai.detect import Detector
from ai.object_trackers import KalmanTracker
import matplotlib.pyplot as plt
from app.utils import configure_logger


logger = configure_logger("appLogger")


class ObjectTrackerApp:
    def __init__(self, mode: str ='single', 
                    target_classes: list=["person"],
                    cache_file: str ="./tmp/cache.json",
                    estimate_acceleration: bool=False,
                    association_metric: str="euclidean"):
        
        assert (mode == 'single') | (mode == 'multi'), f'Unknown mode : {mode}'
        assert isinstance(target_classes, list) , f'Expected target_classes to be list. Got: {type(target_classes)}'
        
        self.mode = mode
        self.tracker = KalmanTracker(mode= self.mode,
                                     association_metric=association_metric,
                                     estimate_acceleration=estimate_acceleration)
        self.target_classes = target_classes
        self.history_cache = {}
        self.cache_size = 100
        self.cache_file = cache_file  # File to store the cache

        # Load existing cache if it exists
        self.load_cache()
    
    
    def process_video(self, video_source:int|str =0)-> None:
        """
        Tracks objects of type target_classes in the provided video
        """ 
        assert isinstance(video_source,(int,str)), f"Expected video_source \
                to be either 0 or a string path. Got: {type(video_source)}"
        try:
            cap = cv2.VideoCapture(video_source)
            
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 1- Detect objects
                boxes = Detector().detect_object(frame) # Assume outputs are [[cls_id, center_x,center_y, w, h]]
                
                # 2- Filter objects
                # Take only boxes that have a class id in the list of tracked objects
                boxes = list(filter(lambda box:  sum([box[0] == class_id\
                                                for class_id in self.target_classes]) > 0,
                                    boxes))
                
                # 3- Update the tracker
                frame, id_estimates = self.tracker.update(boxes, frame)
                
                if id_estimates != None:
                    self.update_cache(id_estimates)
                
                # Display the frame
                cv2.imshow('Object Tracking', frame)
                
                # Break on 'q' key
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

            logger.info("Plotting estimated movement graph per tracked object ...")
            
            destination_figure = "results.png" if video_source == 0 else "movement_"+video_source.split("/")[-1].split(".")[0]+".png"
            self.plot_estimates(destination_figure)
        
        except Exception as e:
            logger.exception(f"Exception during processing video: {e}",exc_info=True)
        
    
    
    
    def plot_estimates(self, dest="object_movement.png"):
        """
        Plots the velocity (vx, vy) for the specified object.
        """
        
        nb_objects = len(self.history_cache)
        
        if self.mode == "single":
            fig, axes = plt.subplots(nb_objects,2, figsize=(8,15))
        else:
            fig, axes = plt.subplots(nb_objects,2,figsize=(15,80))
        
        for object_id in self.history_cache.keys():
            
            velocity = self.history_cache[object_id]['velocity']
            displacement = self.history_cache[object_id]['displacement']
            
            # Extract estimates
            vx = [v[0] for v in velocity]
            vy = [v[1] for v in velocity]
            
            dx = [d[0] for d in displacement]
            dy = [d[1] for d in displacement]
            
            
            if nb_objects >1:
                axes[object_id][0].plot(range(len(vx)),vx, label='vx')
                axes[object_id][0].plot(range(len(vy)),vy, label='vy')
                axes[object_id][0].set_xlabel(f'Time step')
                axes[object_id][0].set_ylabel(f'Velocity')
                axes[object_id][0].set_title(f'Object {object_id +1} Velocity over Time')
                axes[object_id][0].legend()
                
                axes[object_id][1].plot(range(len(dx)), dx, label='dx')
                axes[object_id][1].plot(range(len(dy)),dy, label='dy')
                axes[object_id][1].set_xlabel(f'Time step')
                axes[object_id][1].set_ylabel(f'Displacement')
                axes[object_id][1].set_title(f'Object {object_id +1} Displacement over Time')
                axes[object_id][1].legend()
            else:
                axes[0].plot(range(len(vx)),vx, label='vx')
                axes[0].plot(range(len(vy)),vy, label='vy')
                axes[0].set_xlabel(f'Time step')
                axes[0].set_ylabel(f'Velocity')
                axes[0].set_title(f'Object {object_id +1} Velocity over Time')
                axes[0].legend()
                
                axes[1].plot(range(len(dx)),dx, label='vx')
                axes[1].plot(range(len(dy)),dy, label='vy')
                axes[1].set_xlabel(f'Time step')
                axes[1].set_ylabel(f'Displacement')
                axes[1].set_title(f'Object {object_id +1} Displacement over Time')
                axes[1].legend()
        
        plt.legend()
        # plt.show()    
        # Save the figure
        try:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            
            plt.savefig(os.path.join('./results', dest),dpi=150)
            logger.info(f'Saved the graph to the following path: "./results/{dest}"')
        except Exception as e:
            logger.error(f"Couldn't save the plotted graphs to the path: './results/object_movement.png'\nError: {e}")
            
            
    def update_cache(self, id_estimates_dict):
        """
        Updates the history cache for the specified object and saves it to a JSON file.
        """
        assert isinstance(id_estimates_dict,dict), f'Expected id_estimates_dict to be of type dict. Got: {type(id_estimates_dict)}'
        
        for object_id, (velocity, displacement) in id_estimates_dict.items():
            
            if object_id not in self.history_cache:
                # Initialize deque with max length
                self.history_cache[object_id] = {
                    'velocity': deque(maxlen=self.cache_size),
                    'displacement': deque(maxlen=self.cache_size)
                }

            # Append the new velocity and displacement to the cache
            
            assert len(velocity) == 2 , f"Expected the velocity(vx,vy) to have length 2. Got: {len(velocity)}"
            assert len(displacement) == 2 , f"Expected the velocity(vx,vy) to have length 2. Got: {len(displacement)}"
            
            
            self.history_cache[object_id]['velocity'].append(velocity)
            self.history_cache[object_id]['displacement'].append(displacement)

        # Save the cache to a JSON file
        # self.save_cache()

    def save_cache(self):
        """
        Serializes and saves the cache to a JSON file.
        """
        # Convert deques to lists for JSON serialization
        try:
            serializable_cache = {
                obj_id: {
                    'velocity': list(history['velocity']),
                    'displacement': list(history['displacement'])
                }
                for obj_id, history in self.history_cache.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f)
        except Exception as e:
            logger.exception(f"Exception happened while saving the cache: {e}")
    
    def load_cache(self):
        """
        Loads the cache from a JSON file if it exists.
        """
        try:
            with open(self.cache_file, 'r') as f:
                serializable_cache = json.load(f)

            # Convert lists back to deques
            self.history_cache = {
                obj_id: {
                    'velocity': deque(history['velocity'], maxlen=self.cache_size),
                    'displacement': deque(history['displacement'], maxlen=self.cache_size)
                }
                for obj_id, history in serializable_cache.items()
            }
            logger.info("Cache loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing cache found. Starting fresh.")
            
            # If the parent folder doesn't exist
            # create it
            parent_path = "/".join(self.cache_file.split('/')[:-1])
            if not os.path.exists(parent_path):
                os.makedirs(parent_path)
