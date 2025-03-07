import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# print(dataset)

# step 1
# Ni(n) - the number of times the ad i was selected upto round n
# Ri(n) - the sum of rewards of the ad i up to the round n

N = 500  # total number of users
d = 10  # number of ads the user has clicked on the webpage
ads_selected = []
Ni = [0] * d  # [0, 0, 0, 0..]
Ri = [0] * d  # [0, 0, 0, 0..]
total_rewards = 0

for n in range(1, N):  # iterating with 10,000 users
    ad_no = 0
    max_upper_bound = 0
    for i in range(0, d):  # iterate through the different ads
        if Ni[i] > 0:  # if its already selected
            avg_reward = Ri[i] / Ni[i]
            delta_i = math.sqrt((3 / 2) * math.log(n + 1) / Ni[i])  # +1 since log 0 is infinite
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400   # super high value

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound   # max upper bound
            ad_no = i  # select the ad

    # full list of all the ads selected
    ads_selected.append(ad_no)
    Ni[ad_no] += 1
    # get the reward from dataset
    reward = dataset.values[n, ad_no]
    Ri[ad_no] += reward
    total_rewards += reward

print(ads_selected)
# print(total_rewards)

plt.hist(ads_selected)
plt.title("Histogram of Ads")
plt.xlabel('Ads')
plt.ylabel('Number of times ad was selected')
plt.savefig("Histogram_ads.png")

