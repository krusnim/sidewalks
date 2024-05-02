import matplotlib.pyplot as plt
import geopandas as gp

import nashdata
import bandits
import importlib
import blame
importlib.reload(nashdata)
importlib.reload(bandits)
importlib.reload(blame)

raw = nashdata.RawData("./data")
data = nashdata.SidewalkData(raw)

score = lambda x, y: data.score(x, y)
fair_score = lambda x,y: data.score(x, y, fair_weight=1e-8)
nashville = data.area
validity = lambda x,y: data.is_valid_survey(x, y)

pulls = {}
rewards = {}
payouts = {}

test_bandits = {
    # "fair_gridbandit"      : bandits.GridBandit(nashville, fair_score),
    # "fair_zoombandit"      : bandits.ZoomBandit(nashville, fair_score),
    # "fair_gpbandit"        : bandits.GPBandit(nashville, fair_score),
    # "gridbandit"           : bandits.GridBandit(nashville, score),
    # "zoombandit"           : bandits.ZoomBandit(nashville, score),
    # "gpbandit"             : bandits.GPBandit(nashville, score),
    "prox_gridbandit"      : bandits.GridBandit(nashville, score, validity_check=validity),
    "prox_zoombandit"      : bandits.ZoomBandit(nashville, score, validity_check=validity),
    "prox_gpbandit"        : bandits.GPBandit(nashville, score, validity_check=validity),
    "fair_prox_gridbandit" : bandits.GridBandit(nashville, fair_score, validity_check=validity),
    "fair_prox_zoombandit" : bandits.ZoomBandit(nashville, fair_score, validity_check=validity),
    "fair_prox_gpbandit"   : bandits.GPBandit(nashville, fair_score, validity_check=validity),
}


for name, bandit in test_bandits.items():
    data.clear_hist()
    pulls[name] = []
    rewards[name] = []
    payouts[name] = []
    payout = 0

    print("======================")
    print(f"STARTING \"{name}\" BANDIT")

    for t in range(1000):
        loc, reward = bandit.pull_best()
        payout += reward
        with open(f"./expr/{name}.tsv", "a") as file:
            print("%s\t%.4f\t%.4f" % (str(loc), reward, payout), file=file)
        
        print("T\t%d\tREWARD\t%.4f\t\tPAYOUT\t%.4f" % (t, reward, payout))

        pulls[name] += [loc]
        rewards[name] += [reward]
        payouts[name] += [payout]


fig, ax = plt.subplots()
ax.set_axis_off()
data.df_zip.boundary.plot(ax=ax, color='black')
# gp.GeoSeries(pulls["fair_prox_gridbandit"]).intersection(data.area).plot(color=(0.8, 0.1, 0.1), markersize=6, ax=ax)
# gp.GeoSeries(pulls["fair_prox_zoombandit"]).intersection(data.area).plot(color=(0.82, 0.7, 0.3), markersize=6, ax=ax)
# gp.GeoSeries(pulls["fair_prox_gpbandit"]).intersection(data.area).plot(color=(0.3, 0.5, 1.0), markersize=6, ax=ax)

# data2 = nashdata.SidewalkData(raw)
# data2.city.geometry.intersection(data.area).plot(color=(0.1, 0.6, 0.1), markersize=6, ax=ax)

plt.show()