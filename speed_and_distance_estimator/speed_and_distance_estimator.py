import cv2
import sys
sys.path.append('../')
from utils import distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for obj, obj_tracks in tracks.items():
            if obj == 'ball' or obj == 'referees':
                continue
            frame_numbers = len(obj_tracks)
            for frame_num in range(0, frame_numbers, self.frame_window):
                next_frame = min(frame_num+self.frame_window, frame_numbers-1)

                for track_id, _ in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[next_frame]:
                        continue

                    start_pos = obj_tracks[frame_num][track_id]['position_transformed']
                    end_pos = obj_tracks[next_frame][track_id]['position_transformed']

                    if start_pos is None or end_pos is None:
                        continue
                        
                    distance_covered = distance(start_pos, end_pos)
                    time_elapsed = (next_frame - frame_num) / self.frame_rate
                    speed = distance_covered / time_elapsed # m/s
                    speed = speed * 3.6 # k/h

                    if obj not in total_distance:
                        total_distance[obj] = {}
                    
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0
                    
                    total_distance[obj][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, next_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        
                        tracks[obj][frame_num_batch][track_id]['speed'] = speed
                        tracks[obj][frame_num_batch][track_id]['distance'] = total_distance[obj][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == 'ball' or obj == 'referees':
                    continue
                for _, info in obj_tracks[frame_num].items():
                    if "speed" in info:
                        speed = info.get("speed", None)
                        distance = info.get("distance", None)

                        if speed is None or distance is None:
                            continue

                        bbox = info['bbox']
                        pos = get_foot_position(bbox)
                        pos = list(pos)
                        pos[1] += 40

                        pos = tuple(map(int, pos))
                        cv2.putText(frame, f"{speed:.2f} km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            output_frames.append(frame)

        return output_frames