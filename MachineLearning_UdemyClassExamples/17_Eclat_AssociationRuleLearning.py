import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

# More efficient and scalable version of Apriori.
# Set min support
# Take all subsets in transactions having higher support than min support
# Sort these subsets by decreasing value.

path = r'D:\Udemy\MachineLearning\original\Machine Learning A-Z (Codes and Datasets)\Part 5 - Association Rule Learning\Section 29 - Eclat\Python'

df = pd.read_csv(path+r'\Market_Basket_Optimisation.csv')
# print df.head()

transactions = []
# for i in range(0,len(df)):
for i in range(0,20):
    transactions.append([str(df.values[i,j])for j in range(0,20)])
# print transactions

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 2, min_length = 2)

# Visualising the results
results = list(rules)
# print results
# print(results)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs,rhs,supports))
resultsInDataFrame = pd.DataFrame(inspect(results),columns=['Product','Product 2', 'Support'])

print resultsInDataFrame