"""K-Means zone-density intelligence for AuraTwin AI.

The first feature is the person count detected by CCTV.
"""

import math
import os
import pickle
from typing import List, Dict, Any, Tuple


class KMeansClusterer:
    """
    K-Means clustering implementation for commercial zone thermal-occupancy states.
    Clusters into 3 clusters: 0: Low Density, 1: Medium Density, 2: High Density.
    """

    def __init__(self, n_clusters: int = 3, model_path: str = "models/kmeans_model.pkl"):
        self.n_clusters = n_clusters
        self.model_path = model_path
        self.centroids: List[List[float]] = []
        self.feature_mins: List[float] = []
        self.feature_maxs: List[float] = []
        self.cluster_labels = ["Low Density", "Medium Density", "High Density"]
        self._init_default_centroids()

    def _init_default_centroids(self):
        """Initial baseline centroids [people, temp_c, power_kw, hour_norm, occ_ratio]."""
        self.feature_mins = [0.0, 18.0, 1.0, 0.0, 0.0]
        self.feature_maxs = [120.0, 32.0, 30.0, 24.0, 1.0]

        # Normalized centroids for 3 density stages
        self.centroids = [
            [2.0 / 120.0, (26.0 - 18.0) / 14.0, 2.5 / 30.0, 0.2, 0.05],   # Low density
            [22.0 / 120.0, (24.5 - 18.0) / 14.0, 8.0 / 30.0, 0.5, 0.45],  # Medium density
            [65.0 / 120.0, (23.0 - 18.0) / 14.0, 18.0 / 30.0, 0.6, 0.85], # High density
        ]

    def _normalize(self, features: List[float]) -> List[float]:
        norm = []
        for i, val in enumerate(features):
            f_min = self.feature_mins[i]
            f_max = self.feature_maxs[i]
            denom = max(1e-6, f_max - f_min)
            clipped = max(f_min, min(f_max, val))
            norm.append((clipped - f_min) / denom)
        return norm

    def fit(self, data: List[List[float]], max_iter: int = 100):
        """Fits K-Means centroids on feature vectors."""
        if not data:
            return

        normalized_data = [self._normalize(vec) for vec in data]
        k = min(self.n_clusters, len(normalized_data))

        # Pick initial k spread points
        step = len(normalized_data) // k
        centroids = [normalized_data[i * step] for i in range(k)]

        for _ in range(max_iter):
            clusters: List[List[List[float]]] = [[] for _ in range(k)]
            for vec in normalized_data:
                dists = [self._euclidean(vec, c) for c in centroids]
                closest_idx = dists.index(min(dists))
                clusters[closest_idx].append(vec)

            new_centroids = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    dim = len(cluster[0])
                    mean_vec = [sum(pt[d] for pt in cluster) / len(cluster) for d in range(dim)]
                    new_centroids.append(mean_vec)
                else:
                    new_centroids.append(centroids[i])

            centroids = new_centroids

        # Sort centroids by people/occupancy density
        centroids.sort(key=lambda c: c[0] + c[4])
        self.centroids = centroids
        self.save_model()

    def _euclidean(self, v1: List[float], v2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def predict(self, features: List[float]) -> Tuple[int, str, float]:
        """
        Predict cluster and calculate Zone Density Index (ZDI).
        :param features: [detected_people, temperature, power_kw, time_of_day_hour, occupancy_ratio]
        :return: (cluster_id, cluster_name, zdi_score between 0.0 and 1.0)
        """
        norm_vec = self._normalize(features)
        dists = [self._euclidean(norm_vec, c) for c in self.centroids]
        cluster_id = dists.index(min(dists))

        # Calculate continuous Zone Density Index (ZDI)
        # Weighted combination of occupancy ratio (50%), people count (30%), power load (20%)
        people_score = norm_vec[0]
        occ_score = norm_vec[4]
        power_score = norm_vec[2]

        zdi = round(0.50 * occ_score + 0.30 * people_score + 0.20 * power_score, 3)
        zdi = max(0.0, min(1.0, zdi))

        cluster_name = self.cluster_labels[cluster_id]
        return cluster_id, cluster_name, zdi

    def save_model(self):
        """Persist model state."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "centroids": self.centroids,
                    "feature_mins": self.feature_mins,
                    "feature_maxs": self.feature_maxs,
                    "cluster_labels": self.cluster_labels,
                }, f)
        except Exception as e:
            pass

    def load_model(self):
        """Load model state if exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    state = pickle.load(f)
                    self.centroids = state.get("centroids", self.centroids)
                    self.feature_mins = state.get("feature_mins", self.feature_mins)
                    self.feature_maxs = state.get("feature_maxs", self.feature_maxs)
            except Exception:
                self._init_default_centroids()


kmeans_engine = KMeansClusterer()
