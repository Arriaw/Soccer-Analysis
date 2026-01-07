from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.players_team = {}
        self.kmeans = None

    def get_clustering_model(self, img):
        img_2d = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(img_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        for i, x in enumerate(bbox):
            bbox[i] = int(x)
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = img[0: int(img.shape[0]/2), :]

        kmeans = self.get_clustering_model(img)
        labels = kmeans.labels_
        clustered_img = labels.reshape(img.shape[0], img.shape[1])

        corners = [clustered_img[0][0], clustered_img[0][-1], clustered_img[-1][0], clustered_img[-1][-1]]
        background_label = max(set(corners), key=corners.count)
        player_label = 1 if background_label == 0 else 0
        player_color =  kmeans.cluster_centers_[player_label]

        return player_color

    def assign_team_color(self, frame, players):
        players_color = []

        for _, player in players.items():
            bbox = player['bbox']
            player_color = self.get_player_color(frame, bbox)
            players_color.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(players_color)

        self.kmeans = kmeans

        self.team_colors[0] = kmeans.cluster_centers_[0]
        self.team_colors[1] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.players_team:
            return self.players_team[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        self.players_team[player_id] = team_id

        return team_id
