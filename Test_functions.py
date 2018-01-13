# Test functions for optimization in d=2

def beale(sol):
    # search space = [-4.5,+4.5]
    # sol = (3,0.5)
    return (1.5 - sol[0] + sol[0]*sol[1])**2 + (2.25 - sol[0] + sol[0]*sol[1]**2)**2 + (2.625 - sol[0] + sol[0]*sol[1]**3)**2

def ackley(sol):
    # search space = [-5,5]
    # sol = (0,0)
    return -20*np.exp(-.2*np.sqrt(.5*(sol[0]**2 + sol[1]**2))) - np.exp(.5 * (np.cos(2*np.pi*sol[0]) + np.cos(2*np.pi*sol[1]))) + np.e + 20

def goldstein(sol):
    # search space = [-2,+2]
    # sol = (0,-1)
    return (1 + ((sol[0]+sol[1]+1)**2)*(19-14*sol[0]+3*sol[0]**2 - 14*sol[1]+6*sol[0]*sol[1]+3*sol[1]**2))*(30+((2*sol[0]-3*sol[1])**2)*(18-32*sol[0]+12*sol[0]**2+48*sol[1]-36*sol[0]*sol[1]+27*sol[1]**2))

def booth(sol):
    # search space = [-10,+10]
    # sol = (1,3)
    return (sol[0] + 2*sol[1] - 7)**2 + (2*sol[0]+sol[1]-5)**2

def bukin6(sol):
    # search space = ([-15,-5],[-3,3])
    # sol = (-10,1)
    return 100*np.sqrt(np.abs(sol[1]-0.01*sol[0]**2))+0.01*np.abs(sol[0]+10)

def matyas(sol):
    # search space = [-10,10]
    # sol = (0,0)
    return 0.26*(sol[0]**2+sol[1]**2) - .48*sol[0]*sol[1]

def levi13(sol):
    # search space = [-10,+10]
    # sol = (1,1)
    return np.sin(3*np.pi*sol[0])**2+((sol[0]-1)**2)*(1+np.sin(3*np.pi*sol[1])**2)+((sol[1]-1)**2)*(1+np.sin(2*np.pi*sol[1])**2)

def camel(sol):
    # search space = [-5,+5]
    # sol = (0,0)
    return 2*sol[0]**2 -1.05*sol[0]**4 + (sol[0]**6 / 6) + sol[0]*sol[1] + sol[1]**2

def easom(sol):
    # search space = [-100,+100]
    # sol = (pi,pi)
    return -np.cos(sol[0])*np.cos(sol[1])*np.exp(-((sol[0]-np.pi)**2 + (sol[1]-np.pi)**2))

def cross_in_tray(sol):
    # search space = [-10,+10]
    # sol = (+/- 1.34941,+/- 1.34941)
    return -.0001*(np.abs(np.sin(sol[0])*np.sin(sol[1])*np.exp(np.abs(100-((np.sqrt(sol[0]**2+sol[1]**2))/np.pi))))+1)**.1

def eggholder(sol):
    # search space = [-512,512]
    # sol = (512,404.2319)
    return -(sol[1]+47)*np.sin(np.sqrt(np.abs((sol[0]/2)+sol[1]+47)))-sol[0]*np.sin(np.sqrt(np.abs(sol[0]-(sol[1]+47))))

def holder_table(sol):
    # search space = [-10,+10]
    # sol = (+/- 8.05502, +/- 9.66459)
    return -np.abs(np.sin(sol[0])*np.cos(sol[1])*np.exp(np.abs(1-((np.linalg.norm(sol))/np.pi))))

def mccormick(sol):
    # search space = ([-1.5,+4],[-3,4])
    # sol = (-0.54719,-1.54719)
    return np.sin(sol[0]+sol[1]) + (sol[0]-sol[1])**2 -1.5*sol[0]+2.5*sol[1]+1

def schaffer2(sol):
    # search space = [-100,+100]
    # sol = (0,0)
    return .5 + ((np.sin(sol[0]**2 - sol[1]**2)**2 - .5)/ (1 + .001*(sol[0]**2 + sol[1]**2))**2)


# Test functions for optimization in d >= 2

def rastrigin(sol):
    # search space = [-5.12, 5.12]
    # sol = (0,0,0,...,0)
    A = 10
    return A*len(sol) + np.sum(sol**2 - A*np.cos(2*np.pi*sol))

def sphere(sol):
    # search space = infinite
    # sol = (0,0,0,...,0)
    return np.linalg.norm(sol)**2

def rosenbrock(sol):
    # search space = infinite
    # sol = (1,1,1,...,1)
    return np.sum(100*(sol[1:]-sol[:-1]**2)**2+(sol[:-1]-1)**2)

def styblinski(sol):
    # search space = [-5,5]
    # sol = (-2.903534,...) in [-39.16617n ; -39.16616n]
    return np.sum(sol**4 - 16*sol**2 + 5*sol)/2


# Other functions in low-d
def branin2(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    if x.ndim == 1:
        x = x.reshape(1,-1)
    # search space = [-5, 10], [0, 15]
    # sol = 0.397887 at (-pi, 12.275) or (pi, 2.275) or (9.42478, 2.475)
    return a*(x[:,1] - b*x[:,0]**2 + c*x[:,0] - r)**2 + s*(1-t)*np.cos(x[:,0]) + s

def hartmann3(x):
    if x.ndim == 1:
        x = x.reshape(1,-1)
    alpha = np.array([1,1.2,3,3.2])
    A = np.array([[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]])
    P = (1e-4)*np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])
    res = 0
    for i in range(4):
        temp = 0
        for j in range(3):
            temp -= A[i,j]*(x[:,j]-P[i,j])**2
        res -= alpha[i]*np.exp(temp)
    return res

def hartmann6(x):
    if x.ndim == 1:
        x = x.reshape(1,-1)
    alpha = np.array([1,1.2,3,3.2])
    A = np.array([[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],
                  [3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]])
    P = (1e-4)*np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],
                         [2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    res = 0
    for i in range(4):
        temp = 0
        for j in range(6):
            temp -= A[i,j]*(x[:,j]-P[i,j])**2
        res -= alpha[i]*np.exp(temp)
    return res