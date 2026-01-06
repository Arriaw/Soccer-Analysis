from ultralytics import YOLO
import supervision as sv
import os
import pickle

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        cls_names = detections[0].names # 0:ball 1:goalkeeper 2:player 3:referee
        cls_names_inv = {v:k for k, v in cls_names.items()} 

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame, detection in enumerate(detections):
            # convert to supervision format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # goalkeepr to player
            for i, cls_id in enumerate(detection_sv.class_id):
                if cls_names[cls_id] == 'goalkeeper':
                    detection_sv.class_id[i] = cls_names_inv['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                tracker_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame][tracker_id] = {"bbox":bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame][tracker_id] = {"bbox":bbox}

            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks