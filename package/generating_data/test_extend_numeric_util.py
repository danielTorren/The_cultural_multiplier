import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.optimize import fsolve,least_squares
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
        self.M = individual_params["M"]
        self.A_m = individual_params["A_m"]
        self.P_m = individual_params["P_m"]
        self.q_m = individual_params["q_m"]
        self.lambda_m = individual_params["lambda_m"]
    
    def func_jacobian(self, x, A_0, P_0, q_0, lambda_0, sum_Pq_B):
        # Calculate the Jacobian matrix using approx_fprime
        
        summing_terms = []
        for i in range(self.M):
            term_1 = (((self.A_m[i]*P_0)/(A_0*(self.P_m[i]**(1-self.lambda_m[i]))))**(1/self.lambda_m[i]))
            term = (1/self.lambda_m[i])*term_1*((x + q_0)**((lambda_0/self.lambda_m[i])-1))
            summing_terms.append(term)

        jacobian = lambda_0*(sum(summing_terms))
        return jacobian

    def func_to_solve(self, x, A_0, P_0, q_0,lambda_0, sum_Pq_B):
        #print("x",x)
        #why is x an array?

        #print("chceck", A_0, P_0, q_0,lambda_0)
        #quit()
        #make this faster
        summing_terms = []
        for i in range(self.M):
            term_1 = ((self.A_m[i]*P_0)/(A_0*(self.P_m[i]**(1-self.lambda_m[i]))))**(1/self.lambda_m[i])
            #print("term_1", term_1)
            term = term_1*(x + q_0)**(lambda_0/self.lambda_m[i])
            
            #print("term",term)
            summing_terms.append(term)

        #print("summing_terms", summing_terms,sum(summing_terms))

        f = sum(summing_terms) - sum_Pq_B
        #print("f",f)
        return f

    def calc_consumption_quantities(self):
        root = least_squares(self.func_to_solve, x0=self.init_val, jac=self.func_jacobian, bounds = (0, self.budget/self.P_m[0]),args = (self.A_m[0], self.P_m[0], self.q_m[0],self.lambda_m[0], self.sum_Pq_B))
        #root = fsolve(self.func_to_solve, self.init_val, fprime=self.func_jacobian, args = (self.A_m[0], self.P_m[0], self.q_m[0],self.lambda_m[0], self.sum_Pq_B), bounds=[0])
        
        #print("root",root)
        Q_0 = root["x"][0]
        #print("Q_0",Q_0)
        #print("cost",root["cost"])

        Q_m = self.calc_other_Q(Q_0)
        
        #print("Q_m",Q_m )
        #L_m = (self.eta*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        #L_m = ((self.low_carbon_preferences/(1-self.low_carbon_preferences))*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        
        return Q_m

    def calc_other_Q(self, Q_0):
        #maybe can do this faster
        Q_m = []
        for i in range(self.M):
            Q = (((self.A_m[i]*self.P_m[0])/(self.A_m[0]*self.P_m[i]))**(1/self.lambda_m[i]))*((Q_0 + self.q_m[0])**(self.lambda_m[0]/self.lambda_m[i])) - self.q_m[i]
            Q_m.append(Q)

        return Q_m
    
    def calc_utility(self):
        summing_terms = []
        for i in range(self.M):
            term = self.A_m[i]*((self.Q_m[i] +self.q_m[i])**(1-self.lambda_m[i]))/(1-self.lambda_m[i])
            summing_terms.append(term)

        U = sum(summing_terms)
        return U

    def calc_stuff(self, budget):
        self.budget = budget
        self.init_val = self.budget/self.M
        self.sum_Pq_B = sum(self.P_m*self.q_m) + self.budget

        #print("TEST solutions", self.func_to_solve(self.budget/self.M,self.A_m[0], self.P_m[0], self.q_m[0],self.lambda_m[0], self.sum_Pq_B))
        #print("TEST solutions2 ", self.func_to_solve(self.budget/self.M + 0.01,self.A_m[0], self.P_m[0], self.q_m[0],self.lambda_m[0], self.sum_Pq_B))

        self.Q_m = self.calc_consumption_quantities()
        self.util = self.calc_utility()

        return self.Q_m, self.util
    
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
    "M": 4,
    "A_m": np.asarray([0.02,0.9,0.02,0.06]),
    "P_m": np.asarray([1,1,1,1]),
    "lambda_m": np.asarray([1.3,8.2,4.4,2.5]),
    "q_m": np.asarray([0,0,0,0])
}

"""
params = {
    "M": 3,
    "A_m": np.asarray([0.4,0.8]),
    "P_m": np.asarray([1,1]),
    "lambda_m": np.asarray([2,1.1]),
    "q_m": np.asarray([0,0])
}
"""
b_min = sum(params["P_m"]*params["q_m"])
print("b_min", b_min)
budget_list = np.linspace(b_min,10,100)
#preference_list = np.linspace(0,1,10)

test_subject = Individual_test(params)

data_Q = []
data_U = []
for i in budget_list:
    #print("HEY")
    data_point_Q, data_point_U = test_subject.calc_stuff(i)
    data_Q.append(data_point_Q)
    data_U.append(data_point_U)

data_array_Q = np.asarray(data_Q)
data_array_U = np.asarray(data_U)

data_Q_t = data_array_Q.T
#print("RESULTS Q:",data_array_Q)
#print("U RES:",data_array_U)

fig, ax = plt.subplots()
for i , data in enumerate(data_Q_t):
    ax.plot(budget_list, data, label = "$\lambda$ = %s" % params["lambda_m"][i])
ax.legend()
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Quantity")#row

fig, ax = plt.subplots()
ax.plot(budget_list, data_array_U)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Utility")#row

plt.show()

