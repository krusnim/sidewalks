import geopandas
import numpy as np
import random
from shapely.geometry import Polygon, Point
import torch
import gpytorch

import gp
from utils import polyrand, radiusrand
from powerdiag import powerdiag


class ZoomBandit:
    def __init__(self, area, reward_fn, validity_check=None, init_arms=3):
        self.reward_fn = reward_fn
        self.area = area
        self.t = 1
        self.T = 3000

        self.arms      = []
        self.pull_sum  = []
        self.pull_sum2 = []
        self.n_pulls   = []
        self.t_pulls   = []
        self.validity_check=validity_check
    
        # There have to be at least three arms to begin with or
        #   else the regions can't be computed correctly
        for i in range(init_arms):
            self.arms      += [polyrand(area)]
            self.pull_sum  += [0.1]
            self.pull_sum2 += [0.01]
            self.n_pulls   += [1]
            self.t_pulls   += [1]
        
        # Don't use UCB right away - just use a heuristic
        radii = [area.length/2/init_arms for _ in range(len(self.arms))]
        self.polys = powerdiag(self.arms, radii, self.area)

    def sample_region(self, poly, attempts=10):
        for i in range(attempts):
            p = polyrand(poly) 
            if (not self.validity_check) or self.validity_check(p.x, p.y):
                return p
        return None

    
    def pull(self, arm_i, timeless=False):
        p = self.sample_region(self.polys[arm_i])
        if p is None:
            return None, None


        reward = self.reward_fn(p.x, p.y)
        # print(f"Arm {arm_i} got reward {reward}")

        # Update the arm we just pulled
        self.n_pulls[arm_i] += 1
        self.pull_sum[arm_i] += reward
        self.pull_sum2[arm_i] += reward**2
        if not timeless:
            self.t_pulls[arm_i] = 0
            self.t += 1

        # Poll points from the receding confidence boundary
        #   to determine if new arms need to be made
        new_arm = None
        for loc in self.sample_conf_edge(arm_i):
            if not self.covers(loc):
                new_arm = loc

        # If uncovered space was found, add that point as an arm
        #   and recalculate the polygons that each arm represents
        if new_arm is not None:
            # print("Creating new arm")
            self.arms += [new_arm]
            # self.n_pulls += [self.n_pulls[arm_i]]
            # self.pull_sum += [self.pull_sum[arm_i]]
            # self.pull_sum2 += [self.pull_sum2[arm_i]]
            self.t_pulls += [1]
            self.n_pulls         += [self.n_pulls [arm_i]/2]
            self.n_pulls[arm_i]   =  self.n_pulls [arm_i]/2
            self.pull_sum        += [self.pull_sum[arm_i]/2]
            self.pull_sum[arm_i]  =  self.pull_sum[arm_i]/2
            self.pull_sum2       += [self.pull_sum2[arm_i]/2]
            self.pull_sum2[arm_i] =  self.pull_sum2[arm_i]/2


            radii = [self.ucb(i) for i in range(len(self.arms))]
            self.polys = powerdiag(self.arms, radii, self.area)
            # print(f"{len(radii)} radii + {len(self.arms)} centers => {len(self.polys)} polys")
        
        return p, reward


    def conf_area(self, arm_i, scale=2000, segs=16, margin=0):
        ucb = self.ucb(arm_i)
        return self.arms[arm_i].buffer(scale * ucb + margin, quad_segs=segs/4)
    
    def sample_conf_edge(self, arm_i, scale=2000, margin=0.1, n=20):
        ucb = self.ucb(arm_i)
        r = scale * ucb + margin
        return [radiusrand(self.arms[arm_i], r) for _ in range(n)]
        
    def covers(self, p):
        if not self.area.contains(p):
            return True
        
        for arm_i in range(len(self.arms)):
            area = self.conf_area(arm_i)
            if area is None: 
                continue
            if area.contains(p):
                return True
        return False

    def pull_all(self):
        for i in range(len(self.arms)):
            self.pull(i, timeless=True)
        self.t += 1

    def mean(self, arm_i):
        if self.n_pulls[arm_i] == 0:
            return float("inf")
        return self.pull_sum[arm_i] / self.n_pulls[arm_i]
    
    def var(self, arm_i):
        mean = self.mean(arm_i)
        return self.pull_sum2[arm_i]/self.n_pulls[arm_i] - mean**2
    
    def ucb(self, arm_i):
        return np.sqrt(2*np.log(self.T) / self.n_pulls[arm_i])
        # return np.sqrt(2 / self.n_pulls[arm_i])
    
    def pull_best(self, c=0.3):

        ucb1 = [self.mean(i) + c*self.ucb(i) for i in range(len(self.arms))]
        idxs = sorted(range(len(self.arms)), key=lambda i: ucb1[i], reverse=True)
        
        for idx in idxs:
            p, reward = self.pull(idx)
            if reward is not None:
                return p, reward
        
        return None, 0 # only if no arms are acceptable at all
    
    def to_geodata(self):
        data = {
            "mean": [],
            "arm_i": [],
            "n_pulls": []
        }
        for i in range(len(self.polys)):
            data["mean"]  += [self.mean(i)]
            data["arm_i"] += [i]
            data["n_pulls"] += [self.n_pulls[i]]
        
        return geopandas.GeoDataFrame(data, geometry=self.polys)
    
    def arms_geodata(self):
        data = {
            "mean": [],
            "arm_i": [],
            "n_pulls": []
        }
        for i in range(len(self.arms)):
            data["mean"]  += [self.mean(i)]
            data["arm_i"] += [i]
            data["n_pulls"] += [self.n_pulls[i]]

        areas = [self.conf_area(i) for i in range(len(self.arms))]

        return geopandas.GeoDataFrame(data, geometry=areas)
    
class GridBandit:
    def __init__(self, area, reward_fn, validity_check=None, n=10):
        self.n = n
        xl, yl, xu, yu = area.bounds
        self.xlims = (xl, xu)
        self.ylims = (yl, yu)
        self.reward_fn = reward_fn
        self.valid_mask = np.zeros((n, n))
        self.validity_check = validity_check

        for i in range(n):
            for j in range(n):
                pt = self.poly_for(i, j)
                if area.intersects(pt):
                    self.valid_mask[i, j] = 1 
                

        self.pull_sum =  0.1*np.ones((n, n)) * self.valid_mask
        self.pull_sum2 = 0.1**2*np.ones((n, n)) * self.valid_mask
        self.n_pulls = np.ones((n, n))
        self.t_pulls = np.zeros((n, n))

        self.t = 1

    def sample_region(self, gridx, gridy, attempts=10):
        for i in range(attempts):
            x = random.uniform(
                self.xlims[0] + (gridx)  *(self.xlims[1] - self.xlims[0]) / self.n,
                self.xlims[0] + (gridx+1)*(self.xlims[1] - self.xlims[0]) / self.n,
            )
            y = random.uniform(
                self.ylims[0] + (gridy)  *(self.ylims[1] - self.ylims[0]) / self.n,
                self.ylims[0] + (gridy+1)*(self.ylims[1] - self.ylims[0]) / self.n,
            )
            p = Point(x, y)
            if (not self.validity_check) or self.validity_check(p.x, p.y):
                return p
            
        return None
        
    def pull(self, gridx, gridy, timeless=False):
        p = self.sample_region(gridx, gridy)
        if p is None:
            return None, None

        reward = self.reward_fn(p.x, p.y)
        self.n_pulls[gridx, gridy] += 1
        self.pull_sum[gridx, gridy] += reward
        self.pull_sum2[gridx, gridy] += reward**2

        # print(f"Rewarded {gridx}, {gridy} with {reward}")

        if not timeless:
            self.t_pulls += 1
            self.t_pulls[gridx, gridy] = 0
            self.t += 1

        return p, reward

    def pull_all(self):
        for x in range(self.n):
            for y in range(self.n):
                self.pull(x, y, timeless=True)
        self.t += 1


    def mean(self):
        return (self.pull_sum / self.n_pulls) * self.valid_mask
    
    def var(self):
        mean = self.mean()
        return (self.pull_sum2/self.n_pulls - mean**2)* self.valid_mask
    
    def ucb(self):
        return np.sqrt(2*np.log(self.t) / self.n_pulls) * self.valid_mask
    
    def pull_best(self, c=0.3):
        ucb1 = self.pull_sum / self.n_pulls + c*self.ucb()
        
        idxs = []
        for i in range(self.n):
            for j in range(self.n):
                idxs += [(i, j)]

        idxs = sorted(idxs, key=lambda i: ucb1[i], reverse=True)
        
        for idx in idxs:
            p, reward = self.pull(*idx)
            if reward is not None:
                return p, reward
        
        return None, 0 # only if no arms are acceptable at all
    
    def poly_for(self,gridx, gridy):
        xl = self.xlims[0] + (gridx)  *(self.xlims[1] - self.xlims[0]) / self.n
        xu = self.xlims[0] + (gridx+1)*(self.xlims[1] - self.xlims[0]) / self.n
        yl = self.ylims[0] + (gridy)  *(self.ylims[1] - self.ylims[0]) / self.n
        yu = self.ylims[0] + (gridy+1)*(self.ylims[1] - self.ylims[0]) / self.n
        return Polygon(zip([xl, xl, xu, xu], [yu, yl, yl, yu]))

    
    def to_geodata(self):
        polys = []
        data = {
            "mean": [],
            "gridx": [],
            "gridy": [],
            "n_pulls": []
        }
        mean = self.mean()
        for i in range(self.n):
            for j in range(self.n):
                polys += [self.poly_for(i, j)]
                data["mean"]  += [mean[i, j]]
                data["gridx"] += [i]
                data["gridy"] += [j]
                data["n_pulls"] += [self.n_pulls[i, j]]
        
        return geopandas.GeoDataFrame(data, geometry=polys)
    
    def evaluate(self):
        m = self.mean()
        idx = np.argmax(m)
        gridx, gridy = np.unravel_index(idx, m.shape)
        x, y = self.sample_region(gridx, gridy)
        return self.score(x, y)
    


    
class GPBandit():
    def __init__(self, area, reward_fn, validity_check=None):
        self.n_pulls = 0
        self.x = None
        self.y = None
        self.reward_fn = reward_fn
        self.area = area
        self.lengthscale = 1000
        self.noise = 0.1
        self.validity_check = validity_check
        
        for i in range(3):
            init_pt = polyrand(area)
            self.pull(init_pt.x, init_pt.y)
        self.update_model()

    def pull(self, pos_x, pos_y, update_freq=5):


        if self.validity_check is not None and (not self.validity_check(pos_x, pos_y)):
            return None

        self.n_pulls += 1
        x = torch.Tensor([pos_x, pos_y])[None]
        if self.x is None:
            self.x = x
        else:
            self.x = torch.cat([self.x, x])

        reward = self.reward_fn(pos_x, pos_y)
        # print(f"Reward: {out}")
        y = torch.Tensor([reward])
        if self.y is None:
            self.y = y
        else:
            self.y = torch.cat([self.y, y])
        
        if self.n_pulls % update_freq == 0:
            self.update_model()
        
        return reward

    def pull_best(self, n_samples=20, beta=0.3):
        # https://icml.cc/Conferences/2010/papers/422.pdf

        self.model.eval()
        self.likelihood.eval()

        samples = [polyrand(self.area) for _ in range(n_samples)]

        def get_gpucb(s):
            x = torch.Tensor([s.x, s.y])[None]
            f_preds = self.model(x)
            return f_preds.mean + beta * f_preds.variance
        
        S = sorted(samples, key=get_gpucb, reverse=True)

        for s in S:
            reward = self.pull(s.x, s.y)
            if reward is not None:
                return s, reward
        
        return None, 0 # only if no arms are acceptable at all
    
    def update_model(self, training_iter=50):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = gp.ExactGPModel(self.x, self.y, self.likelihood)
        # print("Updating GP hyperparameters")

        self.model.covar_module.base_kernel.lengthscale = self.lengthscale 
        self.model.covar_module.base_kernel.noise = self.noise

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Reasonable parameter guesses

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model=
            output = self.model(self.x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     self.model.covar_module.base_kernel.lengthscale.item(),
            #     self.model.likelihood.noise.item()
            # ))
            optimizer.step()

        self.lengthscale = self.model.covar_module.base_kernel.lengthscale
        self.noise = self.model.covar_module.base_kernel.noise 

        self.model.eval()
        self.likelihood.eval()
        
    def mean(self, x, y):
        x = torch.Tensor([x, y])[None]
        f_preds = self.model(x)
        return float(f_preds.mean[0])
    
    def var(self, x, y):
        x = torch.Tensor([x, y])[None]
        f_preds = self.model(x)
        return float(f_preds.variance[0])


    def to_geodata(self):
        with torch.no_grad():
            data = {
                # "mean": [],
                # "var": [],
                "zero": []
            }

            locs = [(self.x[i, 0], self.x[i, 1]) for i in range(self.n_pulls)]

            for (x, y) in locs:
                data["zero"] += [0]
            #     data["mean"]  += [self.mean(x, y)]
            #     data["var"] += [self.var(x, y)]

            
            locs = [Point(l) for l in locs]
            
            return geopandas.GeoDataFrame(data, geometry=locs)
