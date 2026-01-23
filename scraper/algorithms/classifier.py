import numpy as np

class PersonalityClassifier:
    def __init__(self):
        self.centroids = {
            'ESTJ': [1, 1, 1, 1], 'ESTP': [1, 1, 1, -1],
            'ESFJ': [1, 1, -1, 1], 'ESFP': [1, 1, -1, -1],
            'ENTJ': [1, -1, 1, 1], 'ENTP': [1, -1, 1, -1],
            'ENFJ': [1, -1, -1, 1], 'ENFP': [1, -1, -1, -1],
            'ISTJ': [-1, 1, 1, 1], 'ISTP': [-1, 1, 1, -1],
            'ISFJ': [-1, 1, -1, 1], 'ISFP': [-1, 1, -1, -1],
            'INTJ': [-1, -1, 1, 1], 'INTP': [-1, -1, 1, -1],
            'INFJ': [-1, -1, -1, 1], 'INFP': [-1, -1, -1, -1],
        }

    def classify(self, user_vector):
        min_dist = float('inf')
        best_match = None
        
        for p_type, centroid in self.centroids.items():
            dist = np.sqrt(sum((u - c) ** 2 for u, c in zip(user_vector, centroid)))
            if dist < min_dist:
                min_dist = dist
                best_match = p_type
                
        return best_match
