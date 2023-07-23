import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.optimize import fsolve
from package.resources.plot import multiline

# modules
class Individual_test:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        individual_params
    ):

        self.low_carbon_preferences = individual_params["low_carbon_preferences"]
        self.prices_low_carbon = individual_params["prices_low_carbon"]
        self.prices_high_carbon = individual_params["prices_high_carbon"]
       
        self.eta = individual_params["eta"]


        self.param_1 = individual_params["param_1"]#their phi
        self.param_2 = individual_params["param_2"]#their lambda

        self.prices_high_carbon_instant = self.prices_high_carbon# + self.carbon_price

        self.Q = self.prices_low_carbon/self.prices_high_carbon_instant

    def func_to_solve(self, x):
        #term_1 = self.prices_low_carbon*((self.prices_high_carbon_instant*self.low_carbon_preferences)/(self.prices_low_carbon*(1-self.low_carbon_preferences)))**(1/self.param_1)
        term_1 = (self.eta*(self.Q**(self.param_1-1)))**(1/self.param_1) 
        term_1 = ((self.low_carbon_preferences/(1-self.low_carbon_preferences))*(self.Q**(self.param_1-1)))**(1/self.param_1)  
        f = term_1*x**(self.param_2/self.param_1) + x - self.budget/self.prices_high_carbon_instant
        return f
    

    def calc_consumption_quantities(self):
        init_vals = 1#  np.asarray([1]*self.M)
        root = fsolve(self.func_to_solve, init_vals)
        #print("root",root)
        H_m = root
        #L_m = (self.eta*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        L_m = ((self.low_carbon_preferences/(1-self.low_carbon_preferences))*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        #L_m = (((self.prices_high_carbon_instant*self.low_carbon_preferences)/(self.prices_low_carbon*(1-self.low_carbon_preferences)))**(1/self.param_1))*H_m**(self.param_2/self.param_2)

        #print("H_m, L_m: ", H_m, L_m)
        
        return H_m,L_m
        #return H_m_clipped,L_m_clipped

    def calc_total_emissions(self):      
        return sum(self.H_m)
    
    def calc_utility(self):
        U = ((1-self.low_carbon_preferences)*self.H_m**(1-self.param_2) + (self.low_carbon_preferences*(1-self.param_2)/(1-self.param_1))*(self.L_m**(1-self.param_1)))**(1/(1-self.param_2))
        #U = (self.H_m**(1-self.param_2) + (self.eta*(1-self.param_2)/(1-self.param_1))*(self.L_m**(1-self.param_1)))**(1/(1-self.param_2))
        return U

    def calc_stuff(self, budget):
        self.budget = budget
        self.H_m, self.L_m = self.calc_consumption_quantities()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.util = self.calc_utility()

        return self.H_m, self.L_m, self.initial_carbon_emissions, self.util
    
def multi_line_matrix_plot( Z, col_vals, row_vals, cmap, col_axis_x, col_label, row_label, y_label):

    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals

    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals
    
    #print("after",xs.shape, ys.shape, c.shape)

    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")
    
    #plt.xticks(x_ticks_pos, x_ticks_label)

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)
    axcb = fig.colorbar(lc)

    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
        #ax.set_xticks(col_ticks_pos)
        #ax.set_xticklabels(col_ticks_label)
        #print("x ticks", col_label,col_ticks_pos, col_ticks_label)
        #ax.set_xlim(left = 0.0, right = 60)
        #ax.set_xlim(left = -10.0, right = 90)

    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")
        #ax.set_xticks(row_ticks_pos)
        #ax.set_xticklabels(row_ticks_label)
        #print("x ticks", row_label, row_ticks_pos, row_ticks_label)
        #ax.set_xlim(left = 1.0)
        #ax.set_xlim(left = 0.0, right = 2.0)

params = {
    "low_carbon_preferences": 0.5,
    "prices_low_carbon": 1,
    "prices_high_carbon": 1,#set q to 1
    "param_1": 1.1,#1.1,#need to GREATER THAN 1 #their phi
    "param_2": 11.5,#need to GREATER THAN param_1#their lambda
    "eta": 75
}

budget_list = np.linspace(0,3,10)
#preference_list = np.linspace(0,1,10)

test_subject = Individual_test(params)

data_H = []
data_L = []
data_E = []
data_U = []
for i in budget_list:
    data_point_H, data_point_L, data_point_E, data_point_U = test_subject.calc_stuff(i)
    data_H.append(data_point_H)
    data_L.append(data_point_L)
    data_E.append(data_point_E)
    data_U.append(data_point_U)

data_array_H = np.asarray(data_H)
data_array_L = np.asarray(data_L)
data_array_E = np.asarray(data_E)
data_array_U = np.asarray(data_U)

fig, ax = plt.subplots()
ax.plot(budget_list, data_array_H, label = "H")
ax.plot(budget_list, data_array_L, label = "L")
ax.legend()
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Quantity")#row

fig, ax = plt.subplots()
ax.plot(budget_list, data_array_H/data_array_L)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Raio H/L")#row

fig, ax = plt.subplots()
ax.plot(budget_list, data_array_E)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Emissions flow")#row

fig, ax = plt.subplots()
ax.plot(budget_list, data_array_U)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Utility")#row

plt.show()

