import numpy as np
from ..models import SkillSignal

class BayesianPredictor:
    def __init__(self):
        self.signals = {}
        db_signals = SkillSignal.objects.all()
        for signal in db_signals:
            self.signals[signal.skill.name] = {
                'P(E|H)': signal.success_rate,
                'P(E|~H)': signal.failure_rate
            }
        self.P_H = 0.10
        self.P_not_H = 0.90

    def predict_success_probability(self, user_skills):
        """
        Calculates P(Hired | Skills) using Naive Bayes.
        """
        if not user_skills:
            return 10.0, []

        log_odds = np.log(self.P_H / self.P_not_H)
        details = []

        for skill in user_skills:
            if skill in self.signals:
                p_e_given_h = self.signals[skill]['P(E|H)']
                p_e_given_not_h = self.signals[skill]['P(E|~H)']
                
                if p_e_given_not_h == 0: p_e_given_not_h = 0.001
                
                lift = p_e_given_h / p_e_given_not_h
                log_odds += np.log(lift)
                
                details.append({
                    'skill': skill,
                    'lift': round(lift, 2),
                    'impact': 'High' if lift > 1.5 else 'Neutral'
                })

        odds = np.exp(log_odds)
        probability = odds / (1 + odds)
        
        return round(probability * 100, 1), details
