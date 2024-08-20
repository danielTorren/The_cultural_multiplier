import numpy as np

if __name__ == '__main__':
    a_preferences, b_preferences = 2, 2
    data = []
    for i in range(1000):
        np.random.seed(i)#For inital construction set a seed
        preferences_beta = np.random.beta(a_preferences, b_preferences, size=6000)
        data.append(np.mean(preferences_beta))
    #print(data)
    print(np.var(data), np.mean(data))
    

