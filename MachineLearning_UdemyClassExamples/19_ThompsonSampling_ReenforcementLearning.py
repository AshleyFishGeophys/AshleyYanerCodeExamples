import pandas as pd
import matplotlib.pyplot as plt
"""
Multi-Armed bandit problem
Distributions
Not deterministic( i.e. UCB). probabilistic
Ad example
Can accomodate delayed feedback
Better empirical evidence
Powerful algorithm
"""

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 6 - Reinforcement Learning\Section 33 - Thompson Sampling\Python'
df = pd.read_csv(path + r'\Ads_CTR_Optimisation.csv')
# print df.head(25)

import random
N = 10000 #Number of people
d = 10 #Number of adds
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward +=reward

print ads_selected

plt.hist(ads_selected)
plt.show()