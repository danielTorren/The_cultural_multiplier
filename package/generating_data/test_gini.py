import numpy as np
import matplotlib.pyplot as plt

def calc_gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def normalize_vector_sum(vec):
    return vec/sum(vec)

N = 200
expenditure_gen_min = 1e-3
expenditure_inequality_const_lin = np.linspace(1,10,20)
expenditure_inequality_const_log = np.logspace(0,1,20)

def calc_gini_expenditure(vect):
    gini_list = []
    for i in vect:
        u = np.linspace(expenditure_gen_min,1,N) #np.random.uniform(size=self.N) #NO LONGER STOCHASTIC
        #print(u,np.random.uniform(size=N))

        no_norm_individual_expenditure_array = u**(-1/i)       
        #no_norm_individual_expenditure_array = np.random.exponential(scale=self.expenditure_inequality_const, size=self.N)
        #print("no_norm_individual_expenditure_array", no_norm_individual_expenditure_array)
        #np.exp(-parameters["individual_expenditure_array_lower"]*np.linspace(parameters["individual_expenditure_array_lower"], parameters["individual_expenditure_array_upper"], num=self.N))
        individual_expenditure_array =  normalize_vector_sum(no_norm_individual_expenditure_array)
        #print("self.individual_expenditure_array", self.individual_expenditure_array,self.expenditure_inequality_const)
        gini = calc_gini(individual_expenditure_array)
        gini_list.append(gini)
    return gini_list

gini_list_lin = calc_gini_expenditure(expenditure_inequality_const_lin)
gini_list_log = calc_gini_expenditure(expenditure_inequality_const_log)
#print("expenditure_inequality_const",expenditure_inequality_const)
#print("gini_list",gini_list)
fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(10,6), sharey=True)
axes[0].scatter(expenditure_inequality_const_lin,  gini_list_lin, color = "red", label = "linear")
axes[1].scatter(expenditure_inequality_const_log,  gini_list_log, color = "blue", label = "log")
axes[0].legend()
axes[1].legend()
axes[0].set_xlabel(r"expenditure inequality const")
axes[1].set_xlabel(r"expenditure inequality const")
axes[0].set_ylabel(r"Gini")
    #print("gini", gini, expenditure_inequality_const)
plt.show()
