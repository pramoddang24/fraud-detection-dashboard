import numpy as np
import random

# Re-defining the MAB classes for joblib to find them correctly
class UCB1:
    def __init__(self, n_arms, alpha=2.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.cumulative_regret = 0.0
    def select_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        return np.argmax(self.values + np.sqrt(self.alpha * np.log(self.t) / (self.counts + 1e-6)))
    def update(self, arm, reward, optimal_reward=1):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward
        self.cumulative_regret += (optimal_reward - reward)

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.cumulative_regret = 0.0
    def select_arm(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))
    def update(self, arm, reward, optimal_reward=1):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.cumulative_regret += (optimal_reward - reward)
