# eclat uses the same apriori model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# print(transactions)

# training the apriori model
from apyori import apriori

# minimal support = Need to have in at least in 3 transactions - 3* 7/ 7501  (3 times daily - 3 * 7 is weekly)
# lift below 3 are not relevant, to keep the lift 3 - 9, (min, max - buy one, get one)
rules = apriori(transactions=transactions, min_support=0.0028, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)

results = list(rules)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

resultsinDataFrameSorted = resultsinDataFrame.nlargest(n=10, columns='Support')

print(resultsinDataFrameSorted)
