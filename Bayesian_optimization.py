# Imports
import GPy
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from scipy.optimize import minimize

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

    def utility(self, x, gp, yMin):
        if self.kind == 'lcb':
            return self._lcb(x, gp, self.kappa)
        if self.kind == 'gp_lcb':
            return self._gp_lcb(x, gp, self.t, self.d, self.delta, self.nu)
        if self.kind == 'ei':
            return self._ei(x, gp, yMin, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, yMin, self.xi)

    @staticmethod
    def _lcb(x, gp, kappa):
        mean, std = gp.predict(x)
        std = np.sqrt(np.clip(std,1e-10,np.inf))
        return mean - kappa * std
    
    @staticmethod
    def _gp_lcb(x, gp, t, d, delta, nu):
        tau_t = lambda x: 2*np.log((x**((d/2)+2))*(np.pi**2) / (3*delta))
        mean, std = gp.predict(x)
        std = np.sqrt(np.clip(std,1e-10,np.inf))
        return mean - np.sqrt(nu*tau_t(t))*std

    @staticmethod
    def _ei(x, gp, yMin, xi):
        mean, std = gp.predict(x)
        std = np.sqrt(np.clip(std,1e-10,np.inf))
        
        f_acq = np.zeros(len(mean)).reshape(-1,1)
        z = (yMin - mean[std != 0] - xi) / std[std != 0]
        f_acq[std != 0] = (yMin - mean[std != 0] - xi) * norm.cdf(z) + std[std != 0] * norm.pdf(z)
        # We return -f_acq because we use a minimizer to solve the opti pb, not a maximizer
        return -f_acq

    @staticmethod
    def _poi(x, gp, yMin, xi):
        mean, std = gp.predict(x)
        std = np.sqrt(np.clip(std,1e-10,np.inf))
        return -norm.cdf((yMin - mean - xi) / std)


class bayesian_optimization:
    """
    Class for performing a Bayesian Optimization task. It is necessary to use the initialize method before using
    the minimize method. This class is made for minimizing an unknown target function, but you can easily maximize
    another function by simply taking the opposite of it.
    
    Input:
        . targetFunc: target function to be minimized. Should be able to take a matrix of size (n_points, dim) as input.
          Examples of correct target functions are given in the functions sphere, rosenbrock, rastrigin and styblinski, given in Test_functions.py
        . lowBounds: the i-th element of the array contains the lower bound for dimension i of the input of
          the target function.
        . highBounds: the i-th element of the array contains the higher bound for dimension i of the input of
          the target function.
        . thres: threshold below which a solution proposed by the algorithm becomes acceptable.
        . dim: dimension of the target function.
        . kernel: name of the kernel to choose. Can be 'Matern52', 'Matern32' or 'RBF'.
        . ARD: logical flag indicating whtether the ARD version of the kernel should be used. Warning: setting
          ARD = True may cause instability in the cholesky decomposition!
        . hedge: True or False. If True, the Hedge version (cf. article "Portfolio Allocation for BO" from Brochu et al.)
          is performed. Note: if True, you must specify a valid dictionary of acquisition functions (cf. below).
        . timeCost: NOT TESTED YET. True or False. If True, performs the time-aware version of BO proposed by Larochelle et al.
          in their paper "Practical BO of ML algorithms".
        . gridSpace: if True, the search space is discretized, and the acquisition function is optimized over it.
        . gridSize: useful only if gridSpace == True. Provides the number of slices in which each dimension is cut.
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
    def __init__(self, targetFunc, lowBounds, highBounds, thres, dim, kernel='Matern52', ARD=False, hedge=False,
                 timeCost=False, gridSpace=False, gridSize=100, normalize_X=True, normalize_y=True, verbose=True):
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
        self.normalize_X = normalize_X
        self.normalize_y = normalize_y
        
        if kernel == 'Matern52':
            self.kernel = GPy.kern.Matern52(self.dim, variance=1., ARD=ARD)
        elif kernel == 'Matern32':
            self.kernel = GPy.kern.Matern32(self.dim, variance=1., ARD=ARD)
        elif kernel == 'RBF':
            self.kernel = GPy.kern.RBF(self.dim, variance=1., ARD=ARD)
        else:
            print('Unknown kernel for the GP. Please choose between "Matern52", "Matern32" and "RBF".')

        
    def initialize(self, nbInitPoints):
        """
        Input:
            . nbInitPoints: number of points initially sampled from the target function.
            
        Note: Many authors recommend to sample at first some random points (for example, dim+1) and get their
        function value, so as to avoid instability in the optimization of the kernel hyperparameters.
        """
        
        self.nbInitPoints = nbInitPoints
        if not self.gridSpace:
            self.startingPoints = np.random.uniform(low=self.lowBounds, high=self.highBounds,
                                                    size=(self.nbInitPoints, self.dim))
        else:
            self.startingPoints = np.array([np.random.choice(np.linspace(self.lowBounds[i],self.highBounds[i],
                                                                         self.gridSize), self.nbInitPoints) for i in range(self.dim)]).T
        self.sampledPoints = self.startingPoints.copy()
        
        if self.normalize_X: # Normalize to [0,1]
            self.sampledPoints = (self.sampledPoints - self.lowBounds) / (self.highBounds - self.lowBounds)
        
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
        if self.normalize_y:
            self.norm_functionValues = (self.functionValues - self.functionValues.mean()) / self.functionValues.std()
        
        if np.min(vals) < self.thres:
            self.done = True
            
        self.bestSolution = self.startingPoints[vals.argmin()]
        if self.normalize_y:
            self.lowestCost = (self.functionValues.min() - self.functionValues.mean()) / self.functionValues.std()
        else:
            self.lowestCost = self.functionValues.min()
        
        self.nbOfQueries += self.nbInitPoints
        self.initialized = True

             
    def minimize(self, maxIter, acqFunc, xi=.01, kappa=2, delta=.1, nu=.2, eta=.01, dicosOfAcq=None,
                 nbInitRand=1e+5, opti_method="L-BFGS-B", nbInitLBFGSB=250, random_method="random",
                 T=1., cooling_schedule="exponential", minTemp=1e-5, discount_fact=.9, grouped=False, loc=False,
                 randCoord=True, minIterSO=1e+2, maxIterSO=1e+5, eps=1e-5, yStar=0, sparse=False, noise_var=1e-10,
                 n_restarts_optim=25):
        """
        Input:
            . maxIter: maximum number of iterations authorized for the main optimization procedure (and not the secondary
            optimizations)
            . acqFunc: acquisition function used for the BO. Can be "lcb", "gp_lcb", "ei" or "poi".
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
            . sparse: logical flag indicating if the sparse version of the Gaussian Process regressor should be used.
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
                noise_var = self.functionValues.var()*0.01
                    
            if not sparse:
                if self.normalize_y:
                    self.gp = GPy.models.GPRegression(self.sampledPoints, self.norm_functionValues.reshape(-1,1),
                                                      kernel=self.kernel, noise_var=noise_var, normalizer=False)
                else:
                    self.gp = GPy.models.GPRegression(self.sampledPoints, self.functionValues.reshape(-1,1),
                                                      kernel=self.kernel, noise_var=noise_var, normalizer=False)
            else:
                if self.normalize_y:
                    self.gp = GPy.models.SparseGPRegression(self.sampledPoints, self.norm_functionValues.reshape(-1,1),
                                                            kernel=self.kernel, num_inducing=10)
                else:
                    self.gp = GPy.models.SparseGPRegression(self.sampledPoints, self.functionValues.reshape(-1,1),
                                                            kernel=self.kernel, num_inducing=10)
            self.gp.Gaussian_noise.constrain_positive(warning=False)
            self.listOfDurGP.append(time.time() - tInitGP)

            #Initialization of the GP for time modelling
            if self.timeCost:
                noise_var_time = self.listOfTimes.var()*0.01
                self.gpForTime = GPy.models.GPRegression(self.sampledPoints, self.listOfTimes.reshape(-1,1),
                                                         kernel=self.kernel, noise_var=noise_var_time)

            self.nbOfIt = 0
            
            if not self.hedge:
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
                # Optimize the acquisition function
                # 2 steps (like in the BO package for Python):
                # First, cheap sample of self.nbInitRand points, and see which one is the lowest
                # Second, real procedure with different possibilities (pure random, L-BFGS-B...)
                
                # First step of optimization
                nominatedPoints = []
                tInitOpti = time.time()
                for ac in listOfAcq:
                    if self.gridSpace == False:
                        xInitRand = np.random.uniform(low=self.lowBounds, high=self.highBounds, size=(int(nbInitRand), self.dim))
                    else:
                        xInitRand = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize), int(nbInitRand)) for l in range(self.dim)]).T
                    
                    if self.normalize_X:
                        xInitRand = (xInitRand - self.lowBounds) / (self.highBounds - self.lowBounds)
                    
                    if self.timeCost:
                        yAcqInitRand = ac.utility(xInitRand, gp=self.gp, yMin=self.lowestCost) / np.exp(self.gpForTime.predict(xInitRand)[0])
                    else:
                        yAcqInitRand = ac.utility(xInitRand, gp=self.gp, yMin=self.lowestCost)
                        
                    yAcqMin = yAcqInitRand.min()
                    xAcqMin = xInitRand[yAcqInitRand.argmin()]
                    
                    # Second step of optimization
                    if opti_method == "L-BFGS-B":
                        xSeeds = np.random.uniform(low=self.lowBounds, high=self.highBounds, size=(int(nbInitLBFGSB),self.dim))
                        if self.normalize_X:
                            xSeeds = (xSeeds - self.lowBounds) / (self.highBounds - self.lowBounds)
                        for xs in xSeeds:
                            if self.timeCost:
                                res = minimize(lambda x: ac.utility(x.reshape(1,-1),
                                                                    gp=self.gp,
                                                                    yMin=self.lowestCost) / np.exp(self.gpForTime.predict(x)[0]),
                                               xs.reshape(1, -1),
                                               bounds = [(self.lowBounds[g], self.highBounds[g]) for g in range(len(xs))],
                                               method = "L-BFGS-B")
                            else:
                                if self.normalize_X:
                                    bounds = [(0,1) for g in range(len(xs))]
                                else:
                                    bounds = [(self.lowBounds[g], self.highBounds[g]) for g in range(self.dim)]
                                    
                                res = minimize(lambda x: ac.utility(x.reshape(1,-1),
                                                                    gp=self.gp,
                                                                    yMin=self.lowestCost),
                                                xs.reshape(1, -1),
                                                bounds = bounds,
                                                method = "L-BFGS-B")

                            if yAcqMin > res.fun[0]:
                                xAcqMin = res.x.copy()
                                yAcqMin = res.fun[0]

                    elif opti_method == "random_search":
                        # We add a random value at the beginning of the list of best costs, so that the following works
                        if not self.gridSpace:
                            randNb = np.random.uniform(low=self.lowBounds, high=self.highBounds, size=self.dim).reshape(1,-1)
                        else:
                            randNb = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize)) for l in range(self.dim)]).T.reshape(1,-1)
                            
                        if self.normalize_X:
                            randNb = (randNb - self.lowBounds) / (self.highBounds - self.lowBounds)
                        
                        listOfCosts = [ac.utility(randNb, gp=self.gp, yMin=self.lowestCost).flatten()[0], yAcqMin]
                        i = 0
                        cptrTemp = 0
                        
                        if random_method in ["metropolis", "simu_anneal"]:
                            currentX = xAcqMin.copy()
                        
                        while ((i < maxIterSO) & (T > minTemp)):
                            if i > minIterSO:
                                if (np.abs(listOfCosts[-1] - listOfCosts[-2]) / np.max([np.abs(listOfCosts[-1]),np.abs(listOfCosts[-2])]) < eps): break
                            for j in range(self.dim):
                                new_sol = xAcqMin.copy()
                                
                                if ((i > maxIterSO) or (T < minTemp)):
                                    break
                                
                                if i > minIterSO:
                                    if (np.abs(listOfCosts[-1] - listOfCosts[-2]) / np.max([np.abs(listOfCosts[-1]),np.abs(listOfCosts[-2])]) < eps): break
                                    
                                if grouped:
                                    gpSize = np.random.randint(self.dim)+1
                                    group = np.random.choice(self.dim, gpSize, replace=False)
                                    
                                    if loc == False:
                                        if self.gridSpace == False:
                                            new_sol[group] = np.random.uniform(low=self.lowBounds[group], high=self.highBounds[group], size=gpSize)
                                        else:
                                            new_sol[group] = np.array([np.random.choice(np.linspace(self.lowBounds[l], self.highBounds[l], self.gridSize)) for l in group]).T
                                    else:
                                        if self.gridSpace == False:
                                            new_sol[group] = np.clip(new_sol[group] + 3 * np.random.randn(gpSize), self.lowBounds[group], self.highBounds[group])
                                        else:
                                            new_sol[group] += np.around(3 * np.random.randn(gpSize)) * minSpace[group]
                                else:
                                    if randCoord:
                                        c = np.random.randint(self.dim)
                                    else:
                                        c = j
                                    
                                    if loc == False:
                                        if self.gridSpace == False:
                                            new_sol[c] = np.random.uniform(low=self.lowBounds[c], high=self.highBounds[c])
                                        else:
                                            new_sol[c] = np.random.choice(np.linspace(self.lowBounds[c], self.highBounds[c], self.gridSize))
                                    else:
                                        if self.gridSpace == False:
                                            new_sol[c] = np.clip(new_sol[c] + 3 * np.random.randn(), self.lowBounds[c], self.highBounds[c])
                                        else:
                                            new_sol[c] += round(3 * np.random.randn()) * minSpace[c]
                                
                                if self.normalize_X:
                                    new_sol = (new_sol - self.lowBounds) / (self.highBounds - self.lowBounds)
                                
                                if random_method == "random":
                                    tempAcqCost = ac.utility(new_sol.reshape(1,-1), gp=self.gp, yMin=self.lowestCost)
                                    if tempAcqCost < yAcqMin:
                                        xAcqMin = new_sol.copy()
                                        yAcqMin = tempAcqCost
                                    listOfCosts.append(tempAcqCost[0])
                                                
                                elif random_method in ["metropolis","simu_anneal"]:
                                    tempAcqCost = ac.utility(new_sol.reshape(1,-1), gp=self.gp, yMin=self.lowestCost)
                                    yAcqCurrent = ac.utility(currentX.reshape(1,-1), gp=self.gp, yMin=self.lowestCost)
                                    alpha = np.min([1, np.exp((yAcqCurrent - tempAcqCost) / T)])
                                    self.listOfAlpha.append(alpha)

                                    if random.random() < alpha:
                                        currentX = new_sol.copy()
                                        self.listOfAcceptance.append(1)
                                    else:
                                        self.listOfAcceptance.append(0)
                                        
                                    if tempAcqCost < ac.utility(xAcqMin.reshape(1,-1), gp=self.gp, yMin=self.lowestCost):
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
                    else:
                        err = "This optimization method has not been implemented yet. Please choose between 'L-BFGS-B' and 'random_search'."
                        raise NotImplementedError(err)
                    nominatedPoints.append(xAcqMin)
                self.listOfDurOpti.append(time.time() - tInitOpti)
                
                if self.hedge:
                    probas = np.exp(eta * self.gains)
                    probas /= np.sum(probas)
                    self.listOfProbas.append(probas)
                    xAcqMin = nominatedPoints[np.random.choice(self.nbOfArms, size=1, p=probas)[0]]
                
                self.sampledPoints = np.vstack((self.sampledPoints,xAcqMin))
                
                t0 = time.time()
                if self.normalize_X:
                    tempVal = self.targetFunc((xAcqMin*(self.highBounds - self.lowBounds) + self.lowBounds).reshape(1,-1))
                else:
                    tempVal = self.targetFunc(xAcqMin.reshape(1,-1))
                
                if self.timeCost:
                    self.listOfTimes.append(time.time() - t0)
                
                self.functionValues = np.append(self.functionValues, tempVal)
                if tempVal == self.functionValues.min():
                    if self.normalize_X:
                        self.bestSolution = xAcqMin*(self.highBounds - self.lowBounds) + self.lowBounds
                    else:
                        self.bestSolution = xAcqMin.copy()
                    
                    if self.normalize_y:
                        self.lowestCost = (tempVal - self.functionValues.mean()) / self.functionValues.std()
                    else:
                        self.lowestCost = tempVal
                
                if self.normalize_y:
                    self.norm_functionValues = (self.functionValues - self.functionValues.mean()) / self.functionValues.std()
                
                self.gapList.append(((self.functionValues.min() - self.functionValues[0]) / (yStar - self.functionValues[0])))
                if self.verbose:
                    print(' ｜ '.join([("%d" % self.nbOfIt).rjust(8),
                                      ("%.2f" % tempVal).rjust(8),
                                      ("%.2f" % self.functionValues.min()).rjust(8),
                                      ("%.2f" % ((self.functionValues.min() - self.functionValues[0]) / (yStar - self.functionValues[0])))]))

                # Update the GP
                t1 = time.time()
                if self.normalize_y:
                    self.gp.set_XY(self.sampledPoints, self.norm_functionValues.reshape(-1,1))
                else:
                    self.gp.set_XY(self.sampledPoints, self.functionValues.reshape(-1,1))
                self.gp.optimize_restarts(num_restarts=n_restarts_optim, max_iters = 1000, verbose=False)
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