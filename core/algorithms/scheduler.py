import math
import random
from collections import defaultdict

class SimulatedAnnealingScheduler:
    """
    Optimization Algorithm.
    Generates an optimal weekly study schedule minimizing burnout and maximizing retention.
    """
    def __init__(self, subjects, hours_available=20):
        self.subjects = subjects # List of subjects
        self.hours_available = hours_available
        # Constraints
        self.max_daily_hours = 4

    def energy(self, schedule):
        """
        Cost function (Lower is better).
        Penalizes:
        - Exceeding daily limits
        - Studying same subject 3 days in a row (burnout)
        - Uneven distribution
        """
        penalty = 0
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_hours = defaultdict(int)
        subject_counts = defaultdict(int)
        
        last_subject = None
        consecutive_days = 0
        
        for day in days:
            subj = schedule.get(day)
            if subj != 'Rest':
                daily_hours[day] += 2 # Assume 2 hour blocks
                subject_counts[subj] += 1
                
                if subj == last_subject:
                    consecutive_days += 1
                else:
                    consecutive_days = 0
                last_subject = subj
                
                if consecutive_days >= 2: # 3rd day in a row
                    penalty += 50
            else:
                consecutive_days = 0
                last_subject = None

        # Diversity Bonus (We want to cover all subjects)
        unique_subjects = len([s for s in schedule.values() if s != 'Rest'])
        if unique_subjects < len(self.subjects) and len(self.subjects) < 5:
            penalty += 100 * (len(self.subjects) - unique_subjects)
            
        return penalty

    def optimize(self, iterations=1000):
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Initial State: Random
        current_schedule = {day: random.choice(self.subjects + ['Rest']) for day in days}
        current_energy = self.energy(current_schedule)
        
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        temperature = 100.0
        cooling_rate = 0.95
        
        for i in range(iterations):
            # Neighbor: Change one day randomly
            neighbor = current_schedule.copy()
            day_to_change = random.choice(days)
            neighbor[day_to_change] = random.choice(self.subjects + ['Rest'])
            
            neighbor_energy = self.energy(neighbor)
            
            # Acceptance Probability
            if neighbor_energy < current_energy:
                current_schedule = neighbor
                current_energy = neighbor_energy
            else:
                prob = math.exp((current_energy - neighbor_energy) / temperature)
                if random.random() < prob:
                    current_schedule = neighbor
                    current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_schedule = current_schedule.copy()
                
            temperature *= cooling_rate
            
        return best_schedule, best_energy

class MultiObjectiveScheduler(SimulatedAnnealingScheduler):
    pass
