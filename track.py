import click
from app.app import ObjectTrackerApp
import os

def load_class_names(filename):
    """Load class names from a file."""
    with open(filename, 'r') as f:
        classes = f.read().splitlines()
    
    classes  = [class_.split(":") for class_ in classes]
    classes = [(int(class_[0]), class_[1].strip()) for class_ in classes]
    id_class = dict(classes)
    
    return id_class



@click.command()
@click.option('--mode', default='single', help='Tracking mode: "single" or "multi".')
@click.option('--video-source', default="./videos/video_senators.mp4", help='Video source (default is "./videos/video_senators.mp4" ). Use 0 for webcam')
@click.option('--show-classes', is_flag=True, help='Display all possible object classes and their IDs.')
@click.option('--target-class', multiple=True, type=int, help='Class(es) to be tracked. Can specify multiple by repeating the flag.')
@click.option('--estimate-acceleration', default=False, type=bool, help='Flag on whether to estimate acceleration of objects. Advisable to set to true for fast moving objects. (default False)')
@click.option('--association-metric', default="euclidean", type=str, help='Metric used to associate objects to tracker: "euclidean" or "mahalanobis". Euclidean is favored when Kalman covariance matrices are not tuned.')

def run_tracking(mode, video_source, show_classes, target_class, estimate_acceleration,association_metric):
    """
    Command-line interface for object tracking with YOLO11 and Kalman filter.
    
    Example usage:
    $ python track.py --mode single --video-source 0
    $ python track.py --show-classes
    $ python track.py --target-class 0 --target-class 1 --target-class 3
    """
    
    if mode != "single":
        mode = "multi"
    print(f"Tracking mode set to: {mode}.")
    
    if video_source != 0:
        assert os.path.exists(video_source), f'Specified video path: {video_source} is not found.'
    
    try:    
        classes = load_class_names("./app/classes.txt")
    except Exception as e:
        print(f'File: "./app/classes.txt" is not Found!')
    
    if show_classes:
        print("Possible object classes and their IDs:\n")
        for id, classname in classes.items():
            print(f"{id}: {classname}")
        return
    
     # Check if target classes were provided
    if len(target_class) == 0:
        print("No target class specified. Tracking objects with class ID: 0.")
        target_class = [0]  # Track all classes
    else:
        if len(target_class) > 1 and mode == "single":
            print(f"Can't specify multiple classes in single object tracking mode!")
            return
        
        for cls_ in target_class:
            assert cls_ < len(classes), f'ID: {cls_} is out of possible classes range'
    
    target_categories = []
    for id_ in target_class:
        target_categories.append(classes[id_])
    
    print(f"Tracking the following categories: {target_categories}")
        
    
    app = ObjectTrackerApp(mode=mode,
                           target_classes=target_categories,
                           estimate_acceleration=estimate_acceleration,
                           association_metric= association_metric)
    app.process_video(video_source=video_source)

if __name__ == '__main__':
    run_tracking()

