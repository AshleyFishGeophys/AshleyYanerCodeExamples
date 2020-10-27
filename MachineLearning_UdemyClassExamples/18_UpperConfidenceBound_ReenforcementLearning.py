import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# i.e. Trains robot dogs how to walk
"""
Multi-Armed Bandit Problem
One-armed bandit is from a slot machine
Regret - non-optimal method
Ad example

Step1: At each round n, we consider two numbers for each ad i
Ni(n) - the number of times the i was selected up to round n
Ri(n) - the sum of rewards of the ad i up to round n.

Step 2: compute the following
- the average reward of ad i up to round n
ri(n) = Ri(n)/Ni(n)
- the confidence interval [ui(n) - deltai(n), ri(n) + deltai(n)] at round n with 
deltai(n) = sqrt(3log(n)/2Ni(n))

Step 3: Select the ad i that has the maximum upper confidence bounds ri(n) + deltai(n) 
"""

# Simulation dataset
path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 6 - Reinforcement Learning\Section 32 - Upper Confidence Bound (UCB)\Python'
df = pd.read_csv(path + r'\Ads_CTR_Optimisation.csv')
# print df.head()

# Upper confidence bound
N = len(df)
d = len(df.columns)
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = df.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

print(sums_of_rewards)
print(ads_selected)

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()