# Imports
import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy
import scipydirect
import random

from scipy.optimize import check_grad
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import norm
import time
from scipy.optimize import minimize

# Kernels
class RBF_:
    def __init__(self, sigma=1, lengthscale=1):
        self.sigma = sigma
        self.lengthscale = lengthscale
    
    def compute(self, X1, X2):
        if X1.ndim == 1:
            X1 = X1.reshape(1,-1)
        if X2.ndim == 1:
            X2 = X2.reshape(1,-1)
        
        res = self.sigma * np.exp(-(.5/self.lengthscale**2) * cdist(X1, X2, metric='sqeuclidean'))
        return res
    
class Matern:
    def __init__(self, nu=2.5, sigma=1, lengthscale=1):
        self.nu = nu
        self.lengthscale = lengthscale
        self.sigma = sigma
        
    def compute(self, X1, X2):
        if X1.ndim == 1:
            X1 = X1.reshape(1,-1)
        if X2.ndim == 1:
            X2 = X2.reshape(1,-1)
            
        if self.nu == 1.5:
            res = self.sigma * (1 + (np.sqrt(3) / self.lengthscale) * cdist(X1, X2, metric='euclidean')) * np.exp(-(np.sqrt(3) / self.lengthscale) * cdist(X1, X2, metric='euclidean'))
            
        if self.nu == 2.5:
            res = self.sigma * (1 + (np.sqrt(5) / self.lengthscale) * cdist(X1, X2, metric='euclidean') + ((5 / (3*self.lengthscale**2)) * cdist(X1, X2, metric='sqeuclidean'))) * np.exp(-(np.sqrt(5) / self.lengthscale) * cdist(X1, X2, metric='euclidean'))
        return res

# Additive GP implementation
class add_gp_regression:
    def __init__(self, decompo, lowBounds, highBounds, normalize_X=True, normalize_y=True,
                 kernel_type = 'rbf', alpha=1e-10):
        self.decompo = decompo
        self.normalize_X = normalize_X
        self.normalize_y = normalize_y
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.lowBounds = lowBounds
        self.highBounds = highBounds
        
        if self.kernel_type == 'rbf':
            self.kernel = RBF_(sigma=1, lengthscale=1)
        elif self.kernel_type == 'matern32':
            self.kernel = Matern(sigma=1, lengthscale=1, nu=1.5)
        elif self.kernel_type == 'matern52':
            self.kernel = Matern(sigma=1, lengthscale=1, nu=2.5)

    
    def loglik(self, theta, X, y, noise_var=None, decompo=None):
        if decompo is None:
            decompo = self.decompo.copy()
        nbOfSamples = X.shape[0]
        dim = X.shape[1]
        if noise_var is None:
            noise_var = self.alpha
        
        K_w_noise = noise_var * np.eye(nbOfSamples)
        if self.kernel_type == 'rbf':
            temp_kernel = RBF_(sigma=theta[0], lengthscale=theta[1])
        elif self.kernel_type == 'matern32':
            temp_kernel = Matern(sigma=theta[0], lengthscale=theta[1], nu=1.5)
        elif self.kernel_type == 'matern52':
            temp_kernel = Matern(sigma=theta[0], lengthscale=theta[1], nu=2.5)

        for i in range(len(decompo)):
            K_w_noise += temp_kernel.compute(X[:,decompo[i]],X[:,decompo[i]])
            
        # Cholesky decomposition to invert the Gram matrix
        L_ = cholesky(K_w_noise, lower=True)
        alpha_ = cho_solve((L_,True), y)
        return -.5*y.dot(alpha_) - np.sum(np.log(np.diag(L_))) - (nbOfSamples/2) * np.log(2*np.pi)
    
    def find_best_decompo(self, d, nbOfTests, X, y, noise_var=None):
        if noise_var is None:
            noise_var = self.alpha
            
        if self.normalize_X:
            X = (X - self.lowBounds) / (self.highBounds - self.lowBounds)
            
        if self.normalize_y:
            mean_y = y.mean()
            std_y = y.std()
            y = (y - mean_y) / std_y
            
        nbOfSamples = X.shape[0]
        dim = X.shape[1]
        
        listOfDecomp = []
        listOfLogLik = []
        for i in range(nbOfTests):
            permut = np.random.permutation(dim).tolist()
            decomp = [permut[x:x+d] for x in range(0,len(permut),d)]
            
            loglik = self.loglik(np.array([self.kernel.sigma, self.kernel.lengthscale]), X, y, noise_var, self.decompo)
            listOfDecomp.append(decomp)
            listOfLogLik.append(loglik)
        self.decompo = listOfDecomp[np.argmax(listOfLogLik)]
        
    def fit(self, X_train, y_train, noise_var=None, n_iter=8, decompo=None, verbose=True):
        
        if self.normalize_X:
            self.X_train = (X_train - self.lowBounds) / (self.highBounds - self.lowBounds)
        else:
            self.X_train = X_train
            
        if self.normalize_y:
            self.mean_y_train = y_train.mean()
            self.std_y_train = y_train.std()
            self.y_train = (y_train - self.mean_y_train) / self.std_y_train
        else:
            self.y_train = y_train
            
        self.nbOfSamples = self.X_train.shape[0]
        if decompo is None:
            decompo = self.decompo.copy()
        if noise_var is None:
            noise_var = self.alpha
        
        tempListOfLogLik = []
        tempListOfTheta = []
        
        # Maximize the log-likelihood to find the shared hyperparameter
        res = scipydirect.minimize(lambda x: -self.loglik(np.exp(x), self.X_train, self.y_train, noise_var, decompo),
                                   bounds=np.log([[1e-5,1e5] for i in range(2)]),
                                   maxT=n_iter)
        if verbose:
            print(res)
            
        sol = res.x
        self.kernel.sigma = np.exp(sol[0])
        self.kernel.lengthscale = np.exp(sol[1])
        
        self.K_w_noise = noise_var * np.eye(self.nbOfSamples)
        for i in range(len(decompo)):
            self.K_w_noise += self.kernel.compute(self.X_train[:,decompo[i]],self.X_train[:,decompo[i]])
        
        self.L_ = cholesky(self.K_w_noise, lower=True)
        self.alpha_ = cho_solve((self.L_,True), self.y_train)
        
        
    def fit_wo_opt(self, X_train, y_train, noise_var=None, decompo=None):
        
        if self.normalize_X:
            self.X_train = (X_train - self.lowBounds) / (self.highBounds - self.lowBounds)
        else:
            self.X_train = X_train
            
        if self.normalize_y:
            self.mean_y_train = y_train.mean()
            self.std_y_train = y_train.std()
            self.y_train = (y_train - self.mean_y_train) / self.std_y_train
        else:
            self.y_train = y_train
            
        self.nbOfSamples = self.X_train.shape[0]
        if decompo is None:
            decompo = self.decompo.copy()
        if noise_var is None:
            noise_var = self.alpha
            
        self.K_w_noise = noise_var * np.eye(self.nbOfSamples)
        for i in range(len(decompo)):
            self.K_w_noise += self.kernel.compute(self.X_train[:,decompo[i]],self.X_train[:,decompo[i]])
        
        self.L_ = cholesky(self.K_w_noise, lower=True)
        self.alpha_ = cho_solve((self.L_,True), self.y_train)
        
    def predict_ind(self, X_test, decompo=None, unnormalize=True):
        if decompo is None:
            decompo = self.decompo.copy()
        
        preds = np.zeros((X_test.shape[0],len(self.decompo),2))
        
        if self.normalize_X:
            X_test = (X_test - self.lowBounds) / (self.highBounds - self.lowBounds)
            
        for i in range(len(decompo)):
            k_star = self.kernel.compute(X_test[:,decompo[i]],self.X_train[:,decompo[i]])
            
            preds[:,i,0] = k_star.dot(self.alpha_)
            
            v=solve_triangular(self.L_, k_star.T, lower=True)
            preds[:,i,1] = np.sqrt(np.clip(np.diag(self.kernel.compute(X_test[:,decompo[i]], X_test[:,decompo[i]]) - v.T.dot(v)), a_min=0, a_max=None))
        
        if (self.normalize_y & unnormalize):
            preds[:,:,0] = preds[:,:,0]*self.std_y_train + self.mean_y_train
            preds[:,:,1] *= np.abs(self.std_y_train)
        
        return preds[:,:,0], preds[:,:,1]
    
    def predict_ind_j(self, X_test, j, decompo=None, unnormalize=True):
        # Warning: this method is made to predict the output only for the GP of group j.
        # As a csq, you need to provide the correct number of columns for X_test.
        if decompo is None:
            decompo = self.decompo.copy()
        
        if self.normalize_X:
            X_test = (X_test - self.lowBounds[decompo[j]]) / (self.highBounds[decompo[j]] - self.lowBounds[decompo[j]])
        
        k_star = self.kernel.compute(X_test, self.X_train[:,decompo[j]])
        means = k_star.dot(self.alpha_)
        
        v=solve_triangular(self.L_, k_star.T, lower=True)
        stds = np.sqrt(np.clip(np.diag(self.kernel.compute(X_test, X_test) - v.T.dot(v)), a_min=0, a_max=None))
        if (self.normalize_y & unnormalize):
            means = means*self.std_y_train + self.mean_y_train
            stds *= np.abs(self.std_y_train)
        
        return means, stds
            
    def predict_combined(self, X_test, decompo=None, unnormalize=True):
        if decompo is None:
            decompo = self.decompo.copy()
        
        preds = np.zeros((X_test.shape[0],2))
        if self.normalize_X:
            X_test = (X_test - self.lowBounds) / (self.highBounds - self.lowBounds)
        
        k_star = np.zeros((len(X_test), self.nbOfSamples))
        k_test = np.zeros(X_test.shape[0])
        
        for i in range(len(decompo)):
            k_star += self.kernel.compute(X_test[:,decompo[i]],self.X_train[:,decompo[i]])
            k_test += np.diag(self.kernel.compute(X_test[:,decompo[i]],X_test[:,decompo[i]]))
            
        preds[:,0] = k_star.dot(self.alpha_)
        
        v=solve_triangular(self.L_, k_star.T, lower=True)
        preds[:,1] = np.sqrt(np.clip(np.diag(k_test - v.T.dot(v)), a_min=0, a_max=None))
        
        if (self.normalize_y & unnormalize):
            preds[:,0] = preds[:,0]*self.std_y_train + self.mean_y_train
            preds[:,1] *= np.abs(self.std_y_train)
        
        return preds[:,0], preds[:,1]

# Functions useful for the schedule of tau
def tau_t_classic(x, d, delta):
    return 2*np.log((x**((d/2)+2))*(np.pi**2) / (3*delta))
def tau_t_kanda(x, d, delta):
    return np.log(2*x)

# Additive Gaussian Process Bayesian Optimization
# Class useful for the acquisition functions
class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, d, delta, t, nu):
        self.kappa = kappa
        self.xi = xi
        self.d = d
        self.delta = delta
        self.t = t
        self.nu = nu

        if kind not in ['lcb', 'ei', 'poi', 'gp_lcb']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of lcb, gp_lcb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, yMin, j, tau_t):
        if self.kind == 'lcb':
            return self._lcb(x, gp, self.kappa, j)
        if self.kind == 'gp_lcb':
            return self._gp_lcb(x, gp, self.t, self.d, self.delta, self.nu, j, tau_t)
        if self.kind == 'ei':
            return self._ei(x, gp, yMin, self.xi, j)
        if self.kind == 'poi':
            return self._poi(x, gp, yMin, self.xi, j)

    @staticmethod
    def _lcb(x, gp, kappa, j):
        means, stds = gp.predict_ind_j(x, j)
        stds = np.clip(stds,1e-10,np.inf)
        return means - kappa * stds
    
    @staticmethod
    def _gp_lcb(x, gp, t, d, delta, nu, j, tau_t=tau_t_kanda):
        means, stds = gp.predict_ind_j(x, j)
        stds = np.clip(stds,1e-10,np.inf)
        return means - np.sqrt(nu*tau_t(t, d, delta))*stds

    @staticmethod
    def _ei(x, gp, yMin, xi, j):
        means, stds = gp.predict_ind_j(x, j)
        stds = np.clip(stds,1e-10,np.inf)
        
        f_acq = np.zeros(len(means))#.reshape(-1,1)
        z = (yMin - means[stds != 0] - xi) / stds[stds != 0]
        f_acq[stds != 0] = (yMin - means[stds != 0] - xi) * norm.cdf(z) + stds[stds != 0] * norm.pdf(z)
        # We return -f_acq because we use a minimizer to solve the opti pb, not a maximizer
        return -f_acq

    @staticmethod
    def _poi(x, gp, yMin, xi, j):
        means, stds = gp.predict_ind_j(x, j)
        stds = np.clip(stds,1e-10,np.inf)
        return -norm.cdf((yMin - means - xi) / stds)

class add_gp_bo:
    """
    Implementation of the Add-GP-UCB algorithm, proposed by Kandasamy et al. in their 2015
    paper.
    It is necessary to use the initialize method before using the minimize method. This class
    is made for minimizing an unknown target function, but you can easily maximize another
    function by simply taking the opposite of it.
    
    Input:
        . targetFunc: target function to be minimized. Should be able to take a matrix of size (n_points, dim) as input.
          Examples of correct target functions are given in the functions sphere, rosenbrock, rastrigin and styblinski.
        . lowBounds: the i-th element of the array contains the lower bound for dimension i of the input of
          the target function.
        . highBounds: the i-th element of the array contains the higher bound for dimension i of the input of
          the target function.
        . thres: threshold below which a solution proposed by the algorithm becomes acceptable.
        . dim: dimension of the target function.
        . kernel: the type of kernel to choose ('rbf', 'matern32' or 'matern52').
        . timeCost: NOT TESTED YET. True or False. If True, performs the time-aware version of BO proposed by Larochelle
          in his paper "Practical BO of ML algorithms". NEEDS TO BE UPDATED.
        . gridSpace: if True, the search space is discretized, and the acquisition function is optimized over it.
        . gridSize: useful only if gridSpace == True. Provides the number of slices in which each dimension is cut.
        . scale_X: logical flag indicating whether to scale the search space to [0,1].
        . normalize_y: logical flag indicating whether to standardize the y or not.
        . verbose: True or False. If True, provides indications for the user at each iteration.
        
    Output:
        . functionValues: list of function values, sampled during the optimization process.
        . done: logical flag. If True, means that the algorithm has reached the optimal zone.
        . nbOfQueries: total number of queries (including the initial ones) used for the optimization procedure.
        . gp: Gaussian Process used to model the target function.
        . gpForTime: only used if timeCost == True. Gaussian Process used to model the log of the time needed to
          compute the target function at each query.
        . startingPoints: the points randomly sampled during the initialization. Attention: this means that we compute
          the function values associated to these points after that.
        . sampledPoints: the points sampled throughout the optimization procedure.
        . bestSolution: best solution found so far (i.e. the one that has the lowest cost)
        . lowestCost: lowest cost found so far.
        . initialized: logical flag indicating if the initialization has been performed or not.
        . listOfDurGP: list of durations for each fitting of the main Gaussian Process.
        . listOfDurOpti: list of durations for each secondary optimization procedure.
        . listOfAlpha: only if opti_method == 'random_search' and random_method == 'metropolis' or 'simu_anneal'. List
          of acceptance probabilities for all secondary optimization procedures, stacked together.
        . listOfAcceptance: only if opti_method == 'random_search' and random_method == 'metropolis' or 'simu_anneal'. List
          of 1 if the point has been accepted and 0 if it has been refused for all secondary optimization procedures,
          stacked together.
        . gains: gains associated with each arm of the bandit. Only if hedge == True.
        . listOfProbas: list of lists of probabilities. Each element of the big list corresponds to an iteration in
          the main optimization procedure, while each element of this last element is the probability to choose arm i.
        . nbOfIt: number of iterations during the main optimization procedure. Different from nbOfQueries.
          
    """
    # The noise in the observations is taken into account by setting the noise_var in the call to the GP reg
    def __init__(self, targetFunc, lowBounds, highBounds, thres, dim, kernel='RBF', hedge=False,
                 timeCost=False, gridSpace=False, gridSize=100, scale_X=True, normalize_y=True,
                 verbose=True):
        self.targetFunc = targetFunc
        self.gapList=[]
        self.lowBounds = lowBounds
        self.highBounds = highBounds
        self.functionValues = []
        self.thres = thres
        self.dim = dim
        self.hedge = hedge
        self.verbose = verbose
        self.done = False
        self.nbOfQueries = 0
        self.gridSpace = gridSpace
        self.gridSize = gridSize
        self.timeCost = timeCost
        self.scale_X = scale_X
        self.normalize_y = normalize_y
        
        # We set the lengthscale to 1e-5 by default to encourage exploration in the beginning
        # of the process (cf. paper)
        if kernel == 'Matern52':
            self.kernel = Matern(nu=2.5, lengthscale=1e-5)
        elif kernel == 'Matern32':
            self.kernel = Matern(nu=1.5, lengthscale=1e-5)
        elif kernel == 'RBF':
            self.kernel = RBF_(lengthscale=1e-5)
        else:
            print('Unknown kernel for the GP. Please choose between "Matern52", "Matern32" and "RBF".')

        
    def initialize(self, nbInitPoints):
        """
        Input:
            . nbInitPoints: number of points initially sampled from the target function.
        """
        
        self.nbInitPoints = nbInitPoints
        if not self.gridSpace:
            self.startingPoints = np.random.uniform(low=self.lowBounds, high=self.highBounds,
                                                    size=(self.nbInitPoints, len(self.lowBounds)))
        else:
            self.startingPoints = np.array([np.random.choice(np.linspace(self.lowBounds[i],self.highBounds[i],
                                                                         self.gridSize), self.nbInitPoints) for i in range(len(self.lowBounds))]).T
        self.sampledPoints = self.startingPoints.copy()
        
        if self.timeCost:
            vals = []
            for i in range(self.nbInitPoints):
                t0 = time.time()
                vals.append(self.targetFunc(self.startingPoints[i]))
                self.listOfTimes.append(time.time()-t0)
            vals=np.array(vals)
            self.listOfTimes = np.array(self.listOfTimes)
        
        else:
            vals = self.targetFunc(self.startingPoints)
        self.functionValues = vals.copy()
        
        if np.min(vals) < self.thres:
            self.done = True
            
        self.bestSolution = self.startingPoints[vals.argmin()]
        self.lowestCost = vals.min()
        
        self.nbOfQueries += self.nbInitPoints
        self.initialized = True

             
    def minimize(self, maxIter, acqFunc, decompo, n_cyc, d, nbOfTests, xi=.01, kappa=2,
                 delta=.1, nu=.2, eta=.01, tau_t=tau_t_kanda, dicosOfAcq=None, nbInitRand=1e+5,
                 opti_method="L-BFGS-B", nbInitLBFGSB=250, random_method="simu_anneal", T=1.,
                 cooling_schedule="exponential", minTemp=1e-5, discount_fact=.9,
                 grouped=False, loc=True, randCoord=True, minIterSO=1e+2, maxIterSO=1e+5,
                 eps=1e-5, yStar=0, noise_var=1e-10, n_iter_DIRECT=8):
        """
        Input:
            . maxIter: maximum number of iterations authorized for the main optimization procedure (and not the secondary
            optimizations)
            . acqFunc: acquisition function used for the BO. Can be "lcb", "gp_lcb", "ei" or "poi".
            . decompo: initial decomposition to provide. List of lists.
            . n_cyc: number of iterations before optimizing the HP of the GP and finding a better
              decomposition.
            . d: size of each group of variables.
            . nbOfTests: number of random permutations of the variables to consider when looking
              for a better decomposition.
            . xi: xi hyperparameter used for the acq functions "ei" and "poi". Governs the trade-off between exploration
              and exploitation. Recommended to fix it at 0.01.
            . kappa: kappa hyperparameter used for the acq function "lcb".
            . delta: delta hyperparameter used for the acq function "gp_lcb". Cf. article 'Portfolio Allocation for BO'
              from Brochu. Usually fixed at 0.1.
            . nu: nu hyperparameter used for the acq function "gp_lcb". Same article than before. Usually fixed at 0.2.
            . eta: learning rate for the GP-Hedge algorithm. Only used if hedge == True in the call to the main class.
            . dicosOfAcq: dictionary of acquisition functions with their hyperparameters, only used for the GP-Hedge
            (i.e. only used if hedge == True) algorithm. It is a list containing a dictionary for each arm of the bandit.
            Each dictionary contains as keys the names of the hyperparameters (+ 'kind') with their different values
            as values.
            . nbInitRand: for the secondary optimization tasks. Number of uniformly random points sampled at the beginning
            of each optimization taks for the acqusition function.
            . opti_method: optimization method for the optimization of the acqusition function. Can only be 'L-BFGS-B'
            or 'random_search'.
            . nbInitLBFGSB: number of multiple starts used for the L-BFGS-B optimization procedure (used to avoid
            getting trapped in local minima, which may be numerous with acqusition functions).
            . random_method: only useful if opti_method == "random_search". Kind of random search performed, can be
              'random', 'metropolis' or 'simu_anneal'.
            . T: initial temperature, useful for 'metropolis' and 'simu_anneal'. Does not change when it is 'metropolis',
            decreases when it is 'simu_anneal'.
            . cooling_schedule: cooling schedule used for the Metropolis-Hastings secondary procedure. For now, only
            'exponential' is available.
            . minTemp: minimal temperature below which the optimization procedure stops when random_method == 'simu_anneal'.
            . discount_fact: factor by which the temperature is multiplied for the simulated annealing optimization.
            . grouped: logical flag indicating if the SO procedure will be performed by groups of variables or not.
            Useful only if opti_method == 'random_search'.
            . loc: logical flag indicating if the new coordinates will be sampled locally or not. Useful only if
            opti_method == 'random_search'.
            . randCoord: logical flag indicating if the new coordinates to update will be chosen randomly or deterministically.
            Useful only if opti_method == 'random_search'.
            . minIterSO: minimum number of iterations for each secondary optimization task.
            . maxIterSO: maximum number of iterations for each secondary optimization task.
            . eps: let us call f_k the acq function value sampled at iteration k of a given secondary optimization procedure.
            Then, if ｜f_k - f_k+1｜/max{｜f_k｜,｜f_k+1｜} < eps, stop the secondary optimization procedure.
            . yStar: minimal value of the objective function. Used to compute the gap metric.
        """
        
        minSpace = (self.highBounds - self.lowBounds) / (self.gridSize - 1)
        self.listOfDurGP = []
        self.listOfDurOpti = []
        
        if random_method in ["metropolis", "simu_anneal"]:
            self.listOfAlpha = []
            self.listOfAcceptance = []
        
        if not self.done:
            if self.hedge:
                self.nbOfArms = len(dicosOfAcq)
                self.gains = np.zeros(self.nbOfArms)
                self.listOfProbas = []

            #First initialization of the GP
            tInitGP = time.time()
            if noise_var is None:
                noise_var = self.functionValues.var()*.01
            
            self.gp = add_gp_regression(decompo, self.lowBounds, self.highBounds,
                                        self.scale_X, self.normalize_y, alpha=noise_var)
            self.gp.kernel = self.kernel
            self.gp.find_best_decompo(d, nbOfTests, self.sampledPoints, self.functionValues,
                                      noise_var)
            self.gp.fit_wo_opt(self.sampledPoints, self.functionValues, noise_var)
            self.listOfDurGP.append(time.time() - tInitGP)

            #Initialization of the GP for time modelling
            if self.timeCost:
                noise_var_time = self.listOfTimes.var()*0.01
                self.gpForTime = GPy.models.GPRegression(self.sampledPoints, self.listOfTimes.reshape(-1,1),
                                                         kernel=self.kernel, noise_var=noise_var_time)

            self.nbOfIt = 0
            
            if self.hedge == False:
                listOfAcq = [UtilityFunction(kind=acqFunc, kappa=kappa, xi=xi, d=self.dim, delta=delta, t=1, nu=nu)]
            else:
                listOfAcq = [UtilityFunction(kind=ac['kind'],
                                             kappa=ac['kappa'],
                                             xi=ac['xi'],
                                             d=self.dim,
                                             delta=ac['delta'],
                                             t=1,
                                             nu=ac['nu']) for ac in dicosOfAcq]

            if self.verbose == True:
                print("Launching BO solver...")
                print(' ｜ '.join([name.center(8) for name in ['Iteration','Objective','Lowest cost','Gap metric']]))
            
            for i in range(int(maxIter)):
                completeXAcqMin = np.zeros(self.dim)
                
                for j in range(len(self.gp.decompo)):
                # Optimize the acquisition function
                # 2 steps (like in the BO package for Python):
                # First, cheap sample of self.nbInitRand points, and see which one is the lowest
                # Second, real procedure with different possibilities (pure random, L-BFGS-B...)
                
                # First step of optimization
                    nominatedPoints = []
                    tInitOpti = time.time()
                    for ac in listOfAcq:
                        if not self.gridSpace:
                            xInitRand = np.random.uniform(low=self.lowBounds[self.gp.decompo[j]], high=self.highBounds[self.gp.decompo[j]], size=(int(nbInitRand), len(self.gp.decompo[j])))
                        else:
                            xInitRand = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize), int(nbInitRand)) for l in range(len(self.lowBounds))]).T

                        if self.timeCost:
                            yAcqInitRand = ac.utility(xInitRand, gp=self.gp, yMin=self.lowestCost) / np.exp(self.gpForTime.predict(xInitRand)[0])
                        else:
                            yAcqInitRand = ac.utility(xInitRand, gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t)
                        
                        yAcqMin = yAcqInitRand.min()
                        xAcqMin = xInitRand[yAcqInitRand.argmin()]

                        # Second step of optimization
                        if opti_method == "L-BFGS-B":
                            xSeeds = np.random.uniform(low=self.lowBounds[self.gp.decompo[j]], high=self.highBounds[self.gp.decompo[j]], size=(int(nbInitLBFGSB),len(self.gp.decompo[j])))
                            for xs in xSeeds:
                                if self.timeCost:
                                    res = minimize(lambda x: ac.utility(x.reshape(1,-1),
                                                                        gp=self.gp,
                                                                        yMin=self.lowestCost) / np.exp(self.gpForTime.predict(x)[0]),
                                                   xs.reshape(1, -1),
                                                   bounds = [(self.lowBounds[g], self.highBounds[g]) for g in range(len(xs))],
                                                   method = "L-BFGS-B")
                                else:
                                    res = minimize(lambda x: ac.utility(x.reshape(1,-1),
                                                                        gp=self.gp,
                                                                        yMin=self.lowestCost,
                                                                        j=j,
                                                                        tau_t=tau_t),
                                                    xs.reshape(1, -1),
                                                    bounds = [(self.lowBounds[g], self.highBounds[g]) for g in self.gp.decompo[j]],
                                                    method = "L-BFGS-B")
                                if yAcqMin > res.fun[0]:
                                    xAcqMin = res.x.copy()
                                    yAcqMin = res.fun[0]

                        elif opti_method == "random_search":
                            # We add a random value at the beginning of the list of best costs, so that the following works
                            if not self.gridSpace:
                                randNb = np.random.uniform(low=self.lowBounds[self.gp.decompo[j]], high=self.highBounds[self.gp.decompo[j]], size=len(self.gp.decompo[j])).reshape(1,-1)
                            else:
                                randNb = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize)) for l in self.gp.decompo[j]]).T.reshape(1,-1)

                            listOfCosts = [ac.utility(randNb, gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t).flatten()[0], yAcqMin]
                            i = 0
                            cptrTemp = 0
                            
                            if random_method in ["metropolis", "simu_anneal"]:
                                currentX = xAcqMin.copy()
                            
                            while ((i < maxIterSO) & (T > minTemp)):
                                if i > minIterSO:
                                    if (np.abs(listOfCosts[-1] - listOfCosts[-2]) / np.max([np.abs(listOfCosts[-1]),np.abs(listOfCosts[-2])]) < eps):
                                        break
                                compteur = 0
                                for l in self.gp.decompo[j]:
                                    new_sol = xAcqMin.copy()
                                    
                                    if ((i > maxIterSO) or (T < minTemp)):
                                        break

                                    if i > minIterSO:
                                        if (np.abs(listOfCosts[-1] - listOfCosts[-2]) / np.max([np.abs(listOfCosts[-1]),np.abs(listOfCosts[-2])]) < eps): break

                                    if grouped:
                                        gpSize = np.random.randint(len(self.gp.decompo[j]))+1
                                        group = np.random.choice(len(self.gp.decompo[j]), gpSize, replace=False)
                                        groupOfIndices = [self.gp.decompo[j][s] for s in group]
                                        
                                        if not loc:
                                            if not self.gridSpace:
                                                new_sol[group] = np.random.uniform(low=self.lowBounds[groupOfIndices], high=self.highBounds[groupOfIndices], size=gpSize)
                                            else:
                                                new_sol[group] = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize)) for l in groupOfIndices]).T
                                        else:
                                            if not self.gridSpace:
                                                new_sol[group] = np.clip(new_sol[group] + 3 * np.random.randn(gpSize), self.lowBounds[groupOfIndices], self.highBounds[groupOfIndices])
                                            else:
                                                new_sol[group] += np.around(3 * np.random.randn(gpSize)) * minSpace[groupOfIndices]
                                    else:
                                        if randCoord:
                                            c = np.random.choice(len(self.gp.decompo[j]))
                                        else:
                                            c = compteur

                                        if loc == False:
                                            if not self.gridSpace:
                                                new_sol[c] = np.random.uniform(low=self.lowBounds[self.gp.decompo[j][c]], high=self.highBounds[self.gp.decompo[j][c]])
                                            else:
                                                new_sol[c] = np.random.choice(np.linspace(self.lowBounds[self.gp.decompo[j][c]], self.highBounds[self.gp.decompo[j][c]], self.gridSize))
                                        else:
                                            if not self.gridSpace:
                                                new_sol[c] = np.clip(new_sol[c] + 3 * np.random.randn(), self.lowBounds[self.gp.decompo[j][c]], self.highBounds[self.gp.decompo[j][c]])
                                            else:
                                                new_sol[c] += round(3 * np.random.randn()) * minSpace[self.gp.decompo[j][c]]

                                    if random_method == "random":
                                        tempAcqCost = ac.utility(new_sol.reshape(1,-1), gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t)
                                        if tempAcqCost < yAcqMin:
                                            xAcqMin = new_sol.copy()
                                            yAcqMin = tempAcqCost
                                        listOfCosts.append(tempAcqCost[0])

                                    elif random_method in ["metropolis","simu_anneal"]:
                                        tempAcqCost = ac.utility(new_sol.reshape(1,-1), gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t)
                                        yAcqCurrent = ac.utility(currentX.reshape(1,-1), gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t)
                                        alpha = np.min([1, np.exp((yAcqCurrent - tempAcqCost) / T)])
                                        self.listOfAlpha.append(alpha)

                                        if random.random() < alpha:
                                            currentX = new_sol.copy()
                                            self.listOfAcceptance.append(1)
                                        else:
                                            self.listOfAcceptance.append(0)

                                        if tempAcqCost < ac.utility(xAcqMin.reshape(1,-1), gp=self.gp, yMin=self.lowestCost, j=j, tau_t=tau_t):
                                            xAcqMin = new_sol.copy()
                                        listOfCosts.append(tempAcqCost[0])

                                        if random_method == "simu_anneal":
                                            cptrTemp += 1
                                            if cptrTemp == 100:
                                                if cooling_schedule == "exponential":
                                                    T *= discount_fact
                                                else:
                                                    err = "This cooling schedule has not been implemented. Please choose among 'exponential'."
                                                    raise NotImplementedError(err)
                                                cptrTemp = 0  
                                    else:
                                        err = "This method for stochastic optimization has not been implemented. Please choose among 'random', 'metropolis' and 'simu_anneal'."
                                        raise NotImplementedError(err)

                                    i += 1
                                    compteur += 1

                        else:
                            err = "This optimization method has not been implemented yet. Please choose between 'L-BFGS-B' and 'random_search'."
                            raise NotImplementedError(err)
                        nominatedPoints.append(xAcqMin)
                    self.listOfDurOpti.append(time.time() - tInitOpti)
                    completeXAcqMin[self.gp.decompo[j]] = xAcqMin
                
                if self.hedge:
                    probas = np.exp(eta * self.gains)
                    probas /= np.sum(probas)
                    self.listOfProbas.append(probas)
                    xAcqMin = nominatedPoints[np.random.choice(self.nbOfArms, size=1, p=probas)[0]]
                
                self.sampledPoints = np.vstack((self.sampledPoints,completeXAcqMin))
                
                t0 = time.time()
                tempVal = self.targetFunc(completeXAcqMin.reshape(1,-1))
                if self.timeCost:
                    self.listOfTimes.append(time.time() - t0)
                    
                if tempVal < self.functionValues.min():
                    self.bestSolution = completeXAcqMin
                    self.lowestCost = tempVal

                self.functionValues = np.append(self.functionValues, tempVal)
                self.gapList.append(((self.lowestCost - self.functionValues[0]) / (yStar - self.functionValues[0])))
                if self.verbose:
                    print(' ｜ '.join([("%d" % self.nbOfIt).rjust(8),
                                     ("%.2f" % tempVal).rjust(8),
                                     ("%.2f" % self.lowestCost).rjust(8),
                                     ("%.2f" % ((self.lowestCost - self.functionValues[0]) / (yStar - self.functionValues[0])))]))

                # Update the GP
                t1 = time.time()
                if self.nbOfIt % n_cyc != 0:
                    self.gp.fit_wo_opt(self.sampledPoints, self.functionValues, noise_var)
                else:
                    self.gp.find_best_decompo(d, nbOfTests, self.sampledPoints, self.functionValues,
                                              noise_var)
                    self.gp.fit(self.sampledPoints, self.functionValues, noise_var,
                                n_iter=n_iter_DIRECT, verbose=False)
                
                self.listOfDurGP.append(time.time() - t1)
                
                if self.timeCost:
                    # Normalize the log(time) ?
                    self.gpForTime.set_XY(self.sampledPoints, np.log(self.listOfTimes).reshape(-1,1))
                    self.gpForTime.optimize_restarts(num_restarts=5, max_iters = 1000, verbose=False)

                if self.hedge:
                    if self.timeCost:
                        rewards = np.array([self.gp.predict(cand.reshape(1,-1))[0] / np.exp(self.gpForTime.predict(cand.reshape(1,-1))[0]) for cand in nominatedPoints])
                    else:
                        rewards = np.array([self.gp.predict(cand.reshape(1,-1))[0] for cand in nominatedPoints])
                    self.gains -= rewards.flatten()

                self.nbOfIt += 1
                self.nbOfQueries += 1

                for ac in listOfAcq:
                    ac.t = self.nbOfIt + 1
                if tempVal < self.thres:
                    if self.verbose:
                        print("Optimization achieved. Acceptable solution found.")
                    self.done = True
                    break
            if ((self.verbose) & (self.done == False)):
                print("Failed to reach the optimal zone.")
        else:
            print("Optimal point found at initialization. No need to keep on searching.")