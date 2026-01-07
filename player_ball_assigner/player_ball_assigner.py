import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox, distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)

        min_dist = 999999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            d1 = distance((player_bbox[0], player_bbox[3]), ball_pos)
            d2 = distance((player_bbox[2], player_bbox[3]), ball_pos)
            d = min(d1, d2)

            if d < min_dist and d < self.max_distance:
                assigned_player = player_id
                min_dist = d

        return assigned_player
    


