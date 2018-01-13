def comp_opt_gap(df, yMin):
    res = (np.minimum.accumulate(df,1) - df[:,0].reshape(df.shape[0],1)) / (yMin - df[:,0].reshape(df.shape[0],1))
    return res

 def compute_means(listOfLists):
    costs = []
    listOfCptr = []
    maxLength = np.max([len(j) for j in listOfLists])
    for i in range(maxLength):
        somme = 0
        cptr = 0
        for liste in listOfLists:
            if i < len(liste):
                somme += liste[i]
                cptr += 1
        costs.append(somme)
        listOfCptr.append(cptr)
    return(np.array(costs) / np.array(listOfCptr))


def acceptance_probability(c_old, c_new, temp, standard = True):
    if standard == True:
        res = np.exp((c_old - c_new) / temp)
    else:
        res = 1 - ((c_new - c_old) / temp)
    return res