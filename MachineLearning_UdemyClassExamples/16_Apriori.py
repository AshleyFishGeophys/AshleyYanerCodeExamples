import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

# Set a minimum support and confidence
# Take all subsets in transactions having higher support than min support
# Take all the rules of these subsets having higher confidence than min
# Sort the rules by decreasing lift.
# Slow due to the amount of combinations.

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 5 - Association Rule Learning\Section 28 - Apriori\Python'
df = pd.read_csv(path + r'\Market_Basket_Optimisation.csv')
# print df.head(25)

transactions = []
# for i in range(0,len(df)):
for i in range(0,1000):
    transactions.append([df.values[i,j] for j in range(0,20)])
    print i

# Training the Apriori model on the dataset.
# Min support - 3 transactions per day*7 days/total (7500)
# Min confidence - try different values
# Nin lift - good at least 3. Below 3 rules are not relevant.
# Min length - 2 products
# Max length - max elements in rule.
# Best deals when buying A you get B. So, in this case 1 product get second free. Depends on business requirements.
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 2, min_length = 2)

# Visualising the results
results = list(rules)
print(results)

results_list =[]
for i in range(0,len(results)):
    results_list.append('RULE:'+ str(results[i][0]) + 'SUPPORRT:' + str(results[i][1]+'INFO:'+str(results[1][2])))
print results_list

# def inspect(results):
#     lhs = [tuple(result[2][0][0][0]) for result in results]
#     rhs = [tuple(result[2][0][1][0]) for result in results]
#     supports = [results[1] for result in results]
#     confidence = [result[2][0][2] for result in results]
#     lifts = [result[2][0][3] for result in results]
#     return list(zip(lhs, rhs, supports, confidence, lifts))
# resultsInDataFrame = pd.DataFrame(inspect(results), columns = [' Left hand side', 'right hand side', 'supports', 'confidence', 'lifts'])
