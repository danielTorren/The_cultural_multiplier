import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from package.resources.utility import get_cmap_colours
from package.resources.utility import check_other_folder
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
        self.a_m = individual_params["a_m"]
        self.A_m = individual_params["A_m"]
        self.P_Hm = individual_params["P_Hm"]
        self.P_Lm = individual_params["P_Lm"]
        self.lambda_m = individual_params["lambda_m"]
        self.sigma_m = individual_params["sigma_m"]
        self.emissions_intensity = individual_params["emissions_intensity"]
        self.psi_m = (self.sigma_m-1)/self.sigma_m
    
    def func_jacobian(self, x, chi_0, psi_0, lambda_0):
        # Calculate the Jacobian matrix using approx_fprime
        
        summing_terms = []
        for i in range(self.M):
            term_1 = (chi_0/self.chi_m[i])**(1/(self.psi_m[i]-self.lambda_m[i]))
            term_2 = self.P_Hm[i] +self.P_Lm[i]*self.Omega_m[i]
            term_3 = ((psi_0-lambda_0)/(self.psi_m[i]-self.lambda_m[i]))

            term = term_1*term_2*term_3*(x)**(((psi_0-lambda_0)/(self.psi_m[i]-self.lambda_m[i]))-1)
            summing_terms.append(term)

        jacobian = sum(summing_terms)

        #print("jacobian",jacobian)
        return jacobian

    def func_to_solve(self, x, chi_0, psi_0, lambda_0):

        summing_terms = []
        for i in range(self.M):
            term_1 = (chi_0/self.chi_m[i])**(1/(self.psi_m[i]-self.lambda_m[i]))
            term_2 = self.P_Hm[i] + self.P_Lm[i]*self.Omega_m[i]

            term = term_1*term_2*(x)**((psi_0-lambda_0)/(self.psi_m[i]-self.lambda_m[i]))
            summing_terms.append(term)

        f = sum(summing_terms) - self.budget

        return f

    def calc_chi_m(self):
        #chi_m = (self.a_m*(1-self.A_m)*self.n_tilde_m**(1-self.lambda_m))/self.P_Hm
        chi_m = (self.a_m*self.n_tilde_m**(1-self.lambda_m))/self.P_Hm
        return chi_m

    def calc_n_tilde_m(self):
        n_tilde_m = (self.A_m*(self.Omega_m**self.psi_m)+(1-self.A_m))**(1/self.psi_m)
        return n_tilde_m
    
    def calc_Omega_m(self):
        Omega_m = ((self.A_m*self.P_Hm)/((1-self.A_m)*self.P_Lm))**(self.sigma_m)
        return Omega_m

    def calc_consumption_quantities(self):
        root = least_squares(self.func_to_solve, x0=self.init_val, jac=self.func_jacobian, bounds = (0, np.inf), args = (self.chi_m[0], self.psi_m[0], self.lambda_m[0]))
        #root = fsolve(self.func_to_solve, self.init_val, fprime=self.func_jacobian, args = (self.A_m[0], self.P_m[0], self.q_m[0],self.lambda_m[0], self.sum_Pq_B), bounds=[0])
        
        #print("root",root)
        H_0 = root["x"][0]
        #print("Q_0",Q_0)
        #print("cost",root["cost"])

        H_m = self.calc_other_H(H_0,self.chi_m[0], self.psi_m[0], self.lambda_m[0])
        L_m = self.Omega_m*H_m
        #print("Q_m",Q_m )
        #L_m = (self.eta*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        #L_m = ((self.low_carbon_preferences/(1-self.low_carbon_preferences))*(H_m**self.param_2)/self.Q)**(1/self.param_1)
        
        return H_m,L_m

    def calc_other_H(self, H_0,chi_0,psi_0,lambda_0):
        #maybe can do this faster
        H_m = []
        for i in range(self.M):
            H = ((chi_0/self.chi_m[i])**(1/(self.psi_m[i]-self.lambda_m[i])))*((H_0)**((psi_0-lambda_0)/(self.psi_m[i]-self.lambda_m[i])))
            H_m.append(H)

        return H_m
    
    def calc_utility(self):
        summing_terms = []
        for i in range(self.M):
            #term = self.A_m[i]*((self.Q_m[i] +self.q_m[i])**(1-self.lambda_m[i]))/(1-self.lambda_m[i])
            term = (self.a_m[i]*((self.H_m[i]*self.n_tilde_m[i])**(1-self.lambda_m[i])))/(1-self.lambda_m[i])
            summing_terms.append(term)

        U = sum(summing_terms)
        return U

    def calc_total_emissions(self):      
        return sum(self.emissions_intensity*self.H_m)

    def calc_stuff(self, budget):
        self.budget = budget
        self.init_val = self.budget/self.M


        self.Omega_m = self.calc_Omega_m()
        self.n_tilde_m = self.calc_n_tilde_m()
        self.chi_m = self.calc_chi_m()

        self.H_m, self.L_m = self.calc_consumption_quantities()
        self.util = self.calc_utility()
        self.E = self.calc_total_emissions()

        return self.H_m, self.L_m , self.E, self.util
"""
params = {
    "M": 4,
    "a_m": np.asarray([0.02,0.9,0.02,0.06]),#[0,1] have to sum to 1
    "A_m": np.asarray([0.5,0.5,0.5,0.5]),#[0,1]
    "P_Hm": np.asarray([0.8,0.8,0.8,0.8]),
    "P_Lm": np.asarray([1,1,1,1]),
    "lambda_m": np.asarray([1.3,8.2,4.4,2.5]),
    "sigma_m": np.asarray([5,5,5,5]),
    "emissions_intensity": np.asarray([1,1,1,1])
}
"""
if __name__ == '__main__' :
    params = {
        "M": 3,
        "a_m": np.asarray([0.1,0.7,0.2]),#[0,1] have to sum to 1
        "A_m": np.asarray([0.5,0.5, 0.5]),#[0,1]
        "P_Hm": np.asarray([0.8,0.8, 0.8]),
        "P_Lm": np.asarray([1,1, 1]),
        "lambda_m": np.asarray([1.5,8, 4]),
        "sigma_m": np.asarray([5,5,5]),
        "emissions_intensity": np.asarray([0.5,5,3])
    }

    budget_list = np.linspace(0,20,100)

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

    #print("data h",data_array_H,data_array_H.shape)

    cmap = get_cmap_colours(params["M"])

    data_array_H_t = data_array_H.T
    data_array_L_t = data_array_L.T
    fig, ax = plt.subplots()
    for i , lambda_m in enumerate(params["lambda_m"]):
        ax.plot(budget_list, data_array_H_t[i], label = "$H,\lambda$ = %s" % lambda_m,color = cmap(i),linestyle='--')
        ax.plot(budget_list, data_array_L_t[i], label = "$L,\lambda$ = %s" % lambda_m,color = cmap(i),linestyle='-')
    ax.legend()
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Quantity")#row
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/quantity_evo_lux_basic" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    fig, ax = plt.subplots()
    ax.plot(budget_list, data_array_U)
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Utility")#row
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/utility_evo_lux_basic" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    fig, ax = plt.subplots()
    ax.plot(budget_list, data_array_E)
    ax.set_xlabel("Budget, B")#col
    #ax.xaxis.set_label_position('top') 
    ax.set_ylabel("Emissions flow")#row
    check_other_folder()
    plotName = "results/Other"
    f = plotName + "/emissions_evo_lux_basic" 
    fig.savefig(f + ".eps", dpi=600, format="eps")
    fig.savefig(f + ".png", dpi=600, format="png") 

    plt.show()

