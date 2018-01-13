# Imports
import numpy as np
import pandas as pd
from scipy.linalg import norm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class random_search:
    def __init__(self):
        self.listOfCosts = []
        self.listOfSols = []
        
    def optimize(self, cost, dim, maxIter, thres, gridSize, minGrid, maxGrid, coord = False,
                 random_coord = False, localized = False, grouped = False, sigma_init = 3,
                 alpha = 0.95):
        """
        Inputs:
        . cost: every cost function possible. Must take the same form than the functions in Test_functions.py.
        . dim: scalar, number of dimensions considered.
        . maxIter: maximum number of iterations allowed.
        . thres: threshold below which we consider that the solution found is acceptable.
        . gridSize: number of lines that slice each dimension.
        . minGrid: minimum value on each dimension.
        . maxGrid: maximum value on each dimension.
        . coord: if True, the coordinate-wise version of the algorithm if performed.
        . random_coord: if True, the coordinates are chosen randomly at each step (Note: only works if coord == True).
        . localized: if True, the proposal points are generated so that they belong to the neighborhood of the previous point.
        . grouped: if True, performs the algorithm group-wise, i.e. we propose to change a group of coordinates at each step.
                   Can be seen as the generalization of coord. Only works if coord = True.
        . sigma_init: only useful if localized == True. Initial standard deviation of the gaussian used to sample new points.
        . alpha: factor by which we multiply the sigma at each iteration. Only useful if localized == True.
        
                   
        Outputs:
        . self.startingPoint: initial point used for the run.
        . self.bestSolutionSoFar: the best solution found by the algorithm. Here, it is also equal to the last solution.
        . self.listOfCosts: list of costs associated with the different solutions sampled. We compute and add a cost only
                            on the best solution so far, this explains why the same cost might be present multiple times
                            in a row.
        . self.lowestCostSoFar: lowest cost of list of costs. Also equals the last cost.
        . self.nbOfIter: number of iterations necessary to reach the optimal zone (or max nb of iterations if the algorithm
                         could not reach the optimal zone).
        . self.optNotReached: logical flag equal to True if the algorithm did not reach the optimal zone.
        
        """
        
        # we assume that all dimensions are sliced in the same fashion (minGrid, maxGrid, gridSize)
        self.startingPoint = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)
        self.bestSolutionSoFar = self.startingPoint.copy()
        
        # Keep track of the proposed points
        self.listOfSols.append(self.startingPoint)
        
        val = cost(self.bestSolutionSoFar)
        self.listOfCosts.append(val)
        
        minSpace = (maxGrid - minGrid) / (gridSize - 1)
        sigma = sigma_init
        i = 0
        
        if coord == True:
            while ((i < maxIter) & (val > thres)):
                for j in range(dim):
                    if ((i > maxIter) or (val < thres)):
                        break
                    new_sol = self.bestSolutionSoFar.copy()
                    
                    if grouped == False:
                        if random_coord == True:
                            c = np.random.randint(dim)
                        else:
                            c = j
                    
                        if localized == False:
                            new_sol[c] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),1)[0]
                        else:
                            new_coord = new_sol[c] + round(np.max((sigma,1)) * np.random.randn()) * minSpace
                            new_sol[c] = new_coord
                            sigma *= alpha
                            
                    else:
                        gpSize = np.random.randint(dim)+1
                        group = np.random.choice(dim, gpSize, replace=False)
                        
                        if localized == False:
                            new_sol[group] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),gpSize)
                            
                        else:
                            new_coord = new_sol[group] + np.around(np.max((sigma,1)) * np.random.randn(gpSize)) * minSpace
                            new_sol[group] = new_coord
                            sigma *= alpha
                    
                    if cost(new_sol) < cost(self.bestSolutionSoFar):
                        self.bestSolutionSoFar = new_sol.copy()
                        val = cost(self.bestSolutionSoFar)
                    self.listOfSols.append(new_sol)
                    self.listOfCosts.append(cost(new_sol))
                              
                    i += 1
        
        else:
            while ((i < maxIter) & (val > thres)):
                if localized == False:
                    new_sol = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)
                else:
                    new_sol = self.bestSolutionSoFar + np.around(np.max((sigma,1)) * np.random.randn(dim)) * minSpace
                    sigma *= alpha

                if cost(new_sol) < cost(self.bestSolutionSoFar):
                    self.bestSolutionSoFar = new_sol.copy()
                    val = cost(self.bestSolutionSoFar)
                self.listOfSols.append(new_sol)
                self.listOfCosts.append(cost(new_sol))    
                                    
                i += 1
        
        self.lowestCostSoFar = np.min(self.listOfCosts)
        self.nbOfIter = i
        
        # Logical flag indicating if the function has reached the optimal ball or not.
        if self.nbOfIter >= maxIter:
            self.optNotReached = True
        else:
            self.optNotReached = False

class grid_search:
    def __init__(self):
        self.listOfCosts = []
        self.listOfSols = []
        
    def optimize(self, cost, dim, maxIter, thres, gridSize, minGrid, maxGrid):
        """
        Inputs:
        . cost: every cost function possible. Must take the same form than the functions already implemented in Test_functions.py.
        . dim: scalar, number of dimensions considered.
        . maxIter: maximum number of iterations allowed.
        . thres: threshold below which we consider that the solution found is acceptable.
        . gridSize: number of lines that slice each dimension.
        . minGrid: minimum value on each dimension.
        . maxGrid: maximum value on each dimension.
                   
        Outputs:
        . self.startingPoint: initial point used for the run.
        . self.bestSolutionSoFar: the best solution found by the algorithm. Here, it is also equal to the last solution.
        . self.nbOfIter: number of iterations necessary to reach the optimal zone (or max nb of iterations if the algorithm
                         could not reach the optimal zone).
        . self.optNotReached: logical flag equal to True if the algorithm did not reach the optimal zone.
        
        """
        # we assume that all dimensions are sliced in the same fashion
        
        n_bins =  gridSize*np.ones(dim)
        bounds = np.repeat([(minGrid,maxGrid)], dim, axis = 0)
        self.setOfPossiblePoints = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]].T.reshape(-1,dim)

        self.startingPoint = self.setOfPossiblePoints[0].copy()
        solution = self.startingPoint.copy()
        self.bestSolutionSoFar = solution.copy()
        self.listOfSols.append(solution)
        
        self.lowestCostSoFar = cost(solution)
        val = cost(solution)
        self.listOfCosts.append(val)

        i = 0
        while ((i < np.min((maxIter,self.setOfPossiblePoints.shape[0]-1))) & (val > thres)):
            solution = self.setOfPossiblePoints[i+1].copy()
            self.listOfSols.append(solution)
            
            val = cost(solution)
            self.listOfCosts.append(val)
            
            if val < self.lowestCostSoFar:
                self.lowestCostSoFar = val
                self.bestSolutionSoFar = solution
            
            i += 1
        
        self.nbOfIter = i
        
        # Logical flag indicating if the function has reached the optimal ball or not.
        if self.nbOfIter >= maxIter:
            self.optNotReached = True
        else:
            self.optNotReached = False

class metropolis_hastings:
    def __init__(self):
        self.listOfCosts = []
        self.listOfAcceptance = []
        self.listOfAlpha = []
    
    def optimize(self, cost, dim, maxIter, thres, gridSize, minGrid, maxGrid, T = 1., sigma_init = 3, discount_factor = .9, coord = False, random_coord = False, grouped = False, indep=False):
        """
        Inputs:
        . cost: every cost function possible. Must take the same form than the functions already implemented in Test_functions.py.
        . dim: scalar, number of dimensions considered.
        . maxIter: maximum number of iterations allowed.
        . thres: threshold below which we consider that the solution found is acceptable.
        . gridSize: number of lines that slice each dimension.
        . minGrid: minimum value on each dimension.
        . maxGrid: maximum value on each dimension.
        . T: constant used in the pi distribution, positive scalar. Controls the rate of acceptance of worse solutions.
        . sigma_init: initial sigma used for the standard deviation of the gaussian used in the localized version.
        . discount_factor: factor by which we multiply the sigma at the end of each iteration.
        . coord: if True, the coordinate-wise version of the algorithm if performed.
        . random_coord: if True, the coordinates are chosen randomly at each step (Note: only works if coord == True)
        . grouped: if True, performs the algorithm group-wise, i.e. we propose to change a group of coordinates at each step.
                   Can be seen as the generalization of coord. Only works if coord = True.
        . indep: if True, the independent version of the MH algorithm is performed. In this version, we sample randomly
                 the new point from the search space. We do not take into account the previous point of the run.
                   
        Outputs:
        . self.startingPoint: initial point used for the run.
        . self.bestSolutionSoFar: the best solution found by the algorithm. Here, it is also equal to the last solution.
        . self.listOfCosts: list of costs associated with the different solutions sampled. We compute and add a cost only
                            on the best solution so far, this explains why the same cost might be present multiple times
                            in a row.
        . self.lowestCostSoFar: lowest cost of list of costs. Also equals the last cost.
        . self.nbOfIter: number of iterations necessary to reach the optimal zone (or max nb of iterations if the algorithm
                         could not reach the optimal zone).
        . self.optNotReached: logical flag equal to True if the algorithm did not reach the optimal zone.
        . self.listOfAlpha: list of the acceptance probabilities computed throughout the run.
        . self.listOfAcceptance: list of logical flags, 1 if the i-th proposed sample had been accepted, 0 otherwise.
        
        """
        self.startingPoint = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)
        solution = self.startingPoint.copy()
        self.bestSolutionSoFar = self.startingPoint.copy()
        self.lowestCostSoFar = cost(self.bestSolutionSoFar)
        val = self.lowestCostSoFar
        self.listOfCosts.append(self.lowestCostSoFar)
        sigma = sigma_init
        
            
        minSpace = (maxGrid - minGrid) / (gridSize - 1)
    
        j = 0
        
        if coord == True:
            while ((j < maxIter) & (val > thres)):
                for d in range(dim):
                    if ((j > maxIter) or (val < thres)):
                        break
                    new_solution = solution.copy()

                    if grouped == False:
                        if random_coord == True:
                            c = np.random.randint(dim) 
                        else:
                            c = d
                            
                        if indep == False:
                            new_solution[c] += round(np.max((sigma,1)) * np.random.randn()) * minSpace
                        else:
                            new_solution[c] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),1)[0]
                            
                    
                    else:
                        gpSize = np.random.randint(dim)+1
                        group = np.random.choice(dim, gpSize, replace=False)
                        
                        if indep == False:
                            new_solution[group] += (np.around(np.max((sigma,1)) * np.random.randn(gpSize)) * minSpace)
                            
                        else:
                            new_solution[group] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),gpSize)
                         
                    
                    alpha = np.min([1,np.exp((cost(solution) - cost(new_solution))/T)])
                    self.listOfAlpha.append(alpha)
                    
                    if random.random() < alpha:
                        solution = new_solution.copy()
                        self.listOfAcceptance.append(1)
                    else:
                        self.listOfAcceptance.append(0)

                    val = cost(solution)
                    self.listOfCosts.append(val)

                    if val < self.lowestCostSoFar:
                        self.lowestCostSoFar = val
                        self.bestSolutionSoFar = solution.copy()
                    
                    j += 1
                    sigma *= discount_factor
        
        else:
            for i in range(int(maxIter)):
                if indep == False:
                    new_solution = solution + np.around(np.max((sigma,1)) * np.random.randn(dim)) * minSpace
                else:
                    new_solution = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)

                # Q symmetric so we do not take it into account
                alpha = np.min((1,np.exp((cost(solution) - cost(new_solution))/T)))
                self.listOfAlpha.append(alpha)
                if random.random() < alpha:
                    solution = new_solution.copy()
                    self.listOfAcceptance.append(1)
                    
                else:
                    self.listOfAcceptance.append(0)

                val = cost(solution)
                self.listOfCosts.append(val)

                if val < self.lowestCostSoFar:
                    self.lowestCostSoFar = val
                    self.bestSolutionSoFar = solution.copy()

                j += 1
                sigma *= discount_factor
                
                if val < thres:
                    break
        
        self.nbOfIter = j
        # Logical flag indicating if the function has reached the optimal ball or not.
        if self.nbOfIter >= maxIter:
            self.optNotReached = True
        else:
            self.optNotReached = False


class simul_anneal_grid:
    def __init__(self):
        self.listOfCosts = []
        self.listOfSols = []
         
    def optimize(self, cost, dim, maxIter, thres, gridSize, minGrid, maxGrid, initTemp = 1., minTemp = 1e-5, alpha = .9, coord = False, rand = False, grouped = False, indep=False, sigma=3, schedule = None, eta=None):
        """
        Inputs:
        . cost: every cost function possible. Must take the same form than the functions already implemented in Test_functions.py.
        . dim: scalar, number of dimensions considered.
        . maxIter: maximum number of iterations allowed.
        . thres: threshold below which we consider that the solution found is acceptable.
        . gridSize: number of lines that slice each dimension.
        . minGrid: minimum value on each dimension.
        . maxGrid: maximum value on each dimension.
        . initTemp: initial temperature used for the SA algorithm.
        . minTemp: minimal temperature allowed. If we do not set it a bit above 0, the algorithm can run indefinitely.
        . alpha: scalar, factor by which we multiply the temperature at each iteration.
        . coord: if True, the coordinate-wise version of the algorithm if performed.
        . rand: if True, the coordinates are chosen randomly at each step (Note: only works if coord == True).
        . grouped: if True, performs the algorithm group-wise, i.e. we propose to change a group of coordinates at each step.
                   Can be seen as the generalization of coord. Only works if coord = True.
        . indep: if True, samples new points uniformly in the search space.
        . sigma: the sigma used for the gaussian in localized/non indep context.
        . schedule: "exponential", "linear" or "turnpike".
        . eta: discount factor, used only for linear schedule.
                   
        Outputs:
        . self.startingPoint: initial point used for the run.
        . self.bestSolutionSoFar: the best solution found by the algorithm. Here, it is also equal to the last solution.
        . self.listOfCosts: list of costs associated with the different solutions sampled. We compute and add a cost only
                            on the best solution so far, this explains why the same cost might be present multiple times
                            in a row.
        . self.lowestCostSoFar: lowest cost of list of costs. Also equals the last cost.
        . self.nbOfIter: number of iterations necessary to reach the optimal zone (or max nb of iterations if the algorithm
                         could not reach the optimal zone).
        . self.optNotReached: logical flag equal to True if the algorithm did not reach the optimal zone.
        
        """
        self.startingPoint = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)
        solution = self.startingPoint.copy()
        self.bestSolutionSoFar = solution.copy()
        self.lowestCostSoFar = cost(solution)
        self.listOfCosts.append(self.lowestCostSoFar)
        
        if schedule == None:
            schedule = "exponential"

        if dim < 5:
            self.listOfSols.append(solution)
            
        old_cost = self.lowestCostSoFar
        minSpace = (maxGrid - minGrid) / (gridSize - 1)
        
        T = initTemp
        j=0
        
        # Note: we can only use grouped == True if rand == True.
        if coord == True:
            while ((T > minTemp) & (j < maxIter)):
                i = 1
                if rand == True:
                    while ((i <= 100) & (j < maxIter)):
                        new_solution = solution.copy()
                        
                        if grouped == False:
                            c = np.random.randint(dim)
                            
                            if indep == False:                         
                                new_solution[c] += round(sigma * np.random.randn()) * minSpace
                            else:
                                new_solution[c] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),1)[0]
                            
                        else:
                            gpSize = np.random.randint(dim)+1
                            group = np.random.choice(dim, gpSize, replace=False)
                            
                            if indep == False:
                                new_solution[group] += np.around(sigma * np.random.randn(gpSize)) * minSpace
                                
                            else:
                                new_solution[group] = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),gpSize)
                         
                        new_cost = cost(new_solution)
                        self.listOfCosts.append(new_cost)
                        ap = acceptance_probability(old_cost, new_cost, T, standard=True)
                        
                        if ap > random.random():
                            solution = new_solution
                            old_cost = new_cost
                            if dim < 5:
                                self.listOfSols.append(solution)
            
                        if old_cost < self.lowestCostSoFar:
                            self.lowestCostSoFar = old_cost
                            self.bestSolutionSoFar = solution    
                    
                        if ((old_cost < thres) or (j > maxIter)):
                            break
       
                        i += 1
                        j += 1
                
                
                else:
                    for d in range(np.max((100,dim))):
                        new_solution = solution.copy()
                        c = d % dim
                        if indep == False:
                            new_solution[c] += round(sigma * np.random.randn()) * minSpace
                        else:
                            new_solution[c]= np.random.choice(np.linspace(minGrid,maxGrid,gridSize),1)[0]
                         
                        new_cost = cost(new_solution)
                        self.listOfCosts.append(new_cost)
                        ap = acceptance_probability(old_cost, new_cost, T, standard=True)
                        
                        if ap > random.random():
                            solution = new_solution
                            old_cost = new_cost
                            if dim < 5:
                                self.listOfSols.append(solution)
            
                        if old_cost < self.lowestCostSoFar:
                            self.lowestCostSoFar = old_cost
                            self.bestSolutionSoFar = solution
                    
                        if ((old_cost < thres) or (j > maxIter)):
                            break
       
                        j += 1
 
                    if ((old_cost < thres) or (j > maxIter)):
                        break
                
                if schedule == "exponential":
                    T *= alpha
                elif schedule == "linear":
                    T -= eta*j
                elif schedule == "turnpike":
                    T = (initTemp - 1) / np.log(j)
                else:
                    print("Unknown cooling strategy. Please choose among 'exponential', 'linear', or 'turnpike'.")
                
                if ((old_cost < thres) or (j > maxIter)):
                        break
        
        else:
            while ((T > minTemp) & (j < maxIter)):
                i = 1
                while i <= 100:
                    if indep == False:
                        new_solution = solution + np.around(np.random.randn(dim)) * minSpace
                    else:
                        new_solution = np.random.choice(np.linspace(minGrid,maxGrid,gridSize),dim)

                    new_cost = cost(new_solution)
                    self.listOfCosts.append(new_cost)
                    ap = acceptance_probability(old_cost, new_cost, T, standard=True)

                    if ap > random.random():
                        solution = new_solution
                        old_cost = new_cost
                        if dim < 5:
                            self.listOfSols.append(solution)

                    if old_cost < self.lowestCostSoFar:
                        self.lowestCostSoFar = old_cost
                        self.bestSolutionSoFar = solution    

                    if ((old_cost < thres) or (j > maxIter)):
                        break

                    i += 1
                    j += 1

                if schedule == "exponential":
                    T *= alpha
                elif schedule == "linear":
                    T -= eta*j
                elif schedule == "turnpike":
                    T = (initTemp - 1) / np.log(j)
                else:
                    print("Unknown cooling strategy. Please choose among 'exponential', 'linear', or 'turnpike'.")

                if ((old_cost < thres) or (j > maxIter)):
                    break
        
        self.nbOfIter = j
        # Logical flag indicating if the function has reached the optimal ball or not.
        if self.lowestCostSoFar > thres:
            self.optNotReached = True
        else:
            self.optNotReached = False