from ultralytics import YOLO
import supervision as sv
import os
import pickle
import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

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
            "players": [],
            "referees": [],
            "ball": []
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
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rec_width, rec_height = 40, 20
        x1_rec = x_center - rec_width // 2
        x2_rec = x_center + rec_width // 2
        y1_rec = (y2 - rec_height // 2) + 15
        y2_rec = (y2 + rec_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rec), int(y1_rec)),
                (int(x2_rec), int(y2_rec)),
                color, 
                cv2.FILLED                
            )

            x1_text = x1_rec + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rec+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-8, y-18],
            [x+8, y-18]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_possession(self, frame, frame_num, team_ball_possession):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_possession_till_now = team_ball_possession[:frame_num+1]
        team1_frames = team_ball_possession_till_now[team_ball_possession_till_now==0].shape[0]
        team2_frames = team_ball_possession_till_now[team_ball_possession_till_now==1].shape[0]

        team1 = team1_frames / (team1_frames + team2_frames)
        team2 = team2_frames / (team1_frames + team2_frames)

        cv2.putText(frame, f"Team 1 Ball Possession: {team1*100:.2f}%", (1375, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Possession: {team2*100:.2f}%", (1375, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotation(self, frames, tracks, team_ball_possession):
        output_video_frames= []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw Referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))


            # Draw Team Ball Possession
            frame = self.draw_team_ball_possession(frame, frame_num, team_ball_possession)

            output_video_frames.append(frame)

        return output_video_frames
    
    def ball_interpolation(self, ball_pos):
        ball_pos = [x.get(1, {}).get("bbox", []) for x in ball_pos]

        df_ball_pos = pd.DataFrame(ball_pos, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_pos = df_ball_pos.interpolate(limit_direction='both')

        ball_pos = [{1: {"bbox": x }} for x in df_ball_pos.to_numpy().tolist()]
        return ball_pos


    def add_positions_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, info in track.items():
                    bbox = info['bbox']
                    if obj == 'ball':
                        pos = get_center_of_bbox(bbox)
                    else:
                        pos = get_foot_position(bbox)
                    tracks[obj][frame_num][track_id]['position'] = pos