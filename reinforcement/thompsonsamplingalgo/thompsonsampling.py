# using the simulation of the same dataset as that of UCB
# USB is a deterministic vs thompson is a probabilistic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# print(dataset.head(10))

import random

N = 500  # these are no of users
d = 10  # number of the ads
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_reward = 0

# looping for the num
for n in range(0, N):
    ad = 0
    highest_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if highest_random < random_beta:
            highest_random = random_beta
            ad = i
    ads_selected.append(ad)

    reward = dataset.values[n, ad]  # get the reward from dataset

    if reward == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1

    total_reward += reward

print("The total reward is: ", total_reward)

plt.hist(ads_selected)
plt.title("Histogram of Ads")
plt.xlabel('Ads')
plt.ylabel('Number of times ad was selected')
plt.savefig("Histogram_ads.png")

