from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/tracks_stubs.pkl')

    # Interpoalte Ball
    tracks['ball'] = tracker.ball_interpolation(tracks['ball'])

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

    # Draw Output
    ## Draw Object Track
    output_video_frames = tracker.draw_annotation(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()