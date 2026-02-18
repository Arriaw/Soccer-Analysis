from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movements import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/tracks_stubs.pkl')

    # Get object positions
    tracker.add_positions_to_tracks(tracks)

    # Camera movement estimator
    cameraMovementEstimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = cameraMovementEstimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')

    cameraMovementEstimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    viewTransformer = ViewTransformer()
    viewTransformer.add_transformed_position_to_tracks(tracks)

    # Interpoalte Ball
    tracks['ball'] = tracker.ball_interpolation(tracks['ball'])


    # Speed And Distance Covered
    speedAndDistanceEstimator = SpeedAndDistanceEstimator()
    speedAndDistanceEstimator.add_speed_and_distance_to_tracks(tracks)


    # # Save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     for i, x in enumerate(bbox): 
    #         bbox[i] = int(x)
    #     frame = video_frames[0]
    #     # crop
    #     cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #     # save
    #     cv2.imwrite(f'output_videos/player_{track_id}.jpg', cropped_image)
    #     break

    # Assign player teams
    teamAssigner = TeamAssigner()
    teamAssigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = teamAssigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = teamAssigner.team_colors[team]

    # Who has the ball
    player_assigner = PlayerBallAssigner()
    team_ball_possession = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_possession) != 0:
                team_ball_possession.append(team_ball_possession[-1])

    team_ball_possession = np.array(team_ball_possession)
    # Draw Output
    ## Draw Object Track
    output_video_frames = tracker.draw_annotation(video_frames, tracks, team_ball_possession)

    ## Draw camera movement
    output_video_frames = cameraMovementEstimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    output_video_frames = speedAndDistanceEstimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()