import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from package.resources.plot import multiline

# modules
class Individual_test:

    """
    Class to represent individuals with identities, preferences and consumption

    """

    def __init__(
        self,
        individual_params,
        low_carbon_preferences,
        sector_preferences,
        H_mins,
        U_mins,
        budget_med,
        omega_m 
    ):

        self.low_carbon_preferences = low_carbon_preferences

        self.sector_preferences = sector_preferences
        
        #self.init_budget = budget
        #self.instant_budget = self.init_budget

        self.carbon_price = individual_params["carbon_price"]

        self.M = individual_params["M"]
        self.sector_substitutability = individual_params["sector_substitutability"]
        self.low_carbon_substitutability_array = individual_params["low_carbon_substitutability"]
        self.omega = (self.sector_substitutability -1)/self.sector_substitutability
        self.psi = (self.low_carbon_substitutability_array -1)/self.low_carbon_substitutability_array

        self.prices_low_carbon = individual_params["prices_low_carbon"]
        self.prices_high_carbon = individual_params["prices_high_carbon"]
        #self.clipping_epsilon = individual_params["clipping_epsilon"]
        self.budget_med = budget_med
        self.u_m_0 = U_mins 

        self.h_m = H_mins
        self.omega_m = omega_m 

        self.prices_high_carbon_instant = self.prices_high_carbon + self.carbon_price

    def calc_omega(self):        
        omega_vector = ((self.prices_high_carbon_instant*self.low_carbon_preferences)/(self.prices_low_carbon*(1- self.low_carbon_preferences )))**(self.low_carbon_substitutability_array)
        return omega_vector
    
    def calc_tilde(self):
        tilde_n_m = (self.low_carbon_preferences*(self.Omega_m**(self.psi)) + (1-self.low_carbon_preferences))**(1/self.psi)
        return tilde_n_m
    
    def calc_chi(self):
        chi_array = (self.prices_high_carbon_instant/(self.sector_preferences*(self.tilde_n_m**(self.omega))))**(1/(self.omega-1))
        return chi_array


    def calc_consumption_quantities_cobbs_min_u_prop_B(self):

        sum_numerator = sum((self.u_m_0/self.tilde_n_m)*(self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant))
        sum_denominator = sum((self.omega_m/self.prices_high_carbon_instant)*(self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant))
        term_1 = (self.omega_m*self.budget_med - self.omega_m*sum_numerator)/(self.prices_high_carbon_instant*sum_denominator)
        term_2 = (self.u_m_0*self.prices_high_carbon_instant)/(self.tilde_n_m*self.omega_m)
        H_m = (self.instant_budget/self.budget_med)*(term_1 + term_2)
        L_m = self.Omega_m*H_m
        return H_m, L_m
        #return H_m_clipped,L_m_clipped

    def calc_consumption_quantities(self):
        common_vector_denominator = sum(self.chi_m*(self.Omega_m*self.prices_low_carbon + self.prices_high_carbon_instant))        

        H_m = self.h_m + self.chi_m*(self.instant_budget - sum(self.h_m*self.prices_high_carbon_instant))/common_vector_denominator

        L_m = self.Omega_m*self.chi_m*(self.instant_budget - sum(self.h_m*self.prices_high_carbon_instant))/common_vector_denominator

        #H_m_clipped = np.clip(H_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        #L_m_clipped = np.clip(L_m, 0 + self.clipping_epsilon, 1- self.clipping_epsilon)
        
        return H_m,L_m
        #return H_m_clipped,L_m_clipped

    def calc_total_emissions(self):      
        return sum(self.H_m)

    def calc_utility_cobbs_min_u_prop_B(self):

        psuedo_utility = (self.low_carbon_preferences*(self.L_m**(self.psi)) + (1 - self.low_carbon_preferences)*(self.H_m**(self.psi)))**(1/self.psi)
        
        inside_prod_u  = (psuedo_utility - ((self.instant_budget*self.u_m_0)/self.budget_med))**(self.omega) #i dont think i need to re_do the derivation as u_m is still a constant not variable
        U = np.product(inside_prod_u)
        return U
    
    def calc_utility(self):

        psuedo_utility = (self.low_carbon_preferences*(self.L_m**(self.psi)) + (1 - self.low_carbon_preferences)*((self.H_m - self.h_m)**(self.psi)))**(1/self.psi)
        
        sum_U = (sum(self.sector_preferences*(psuedo_utility**(self.omega))))**(1/self.omega)
        return sum_U

    def calc_stuff_cobbs_min_u_prop_B(self, budget):

        self.instant_budget = budget
        self.Omega_m = self.calc_omega()
        self.tilde_n_m = self.calc_tilde()
        #self.chi_m = self.calc_chi()

        self.H_m, self.L_m = self.calc_consumption_quantities_cobbs_min_u_prop_B()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.utility = self.calc_utility_cobbs_min_u_prop_B()

        print("self.H_m",self.H_m)
        #print("self.L_m",self.L_m)
        #print("self.initial_carbon_emissions", self.initial_carbon_emissions)
        #print("self.utility", self.utility)

        return self.initial_carbon_emissions, self.utility
    
    def calc_stuff(self, budget):

        self.instant_budget = budget
        self.Omega_m = self.calc_omega()
        self.tilde_n_m = self.calc_tilde()
        self.chi_m = self.calc_chi()

        self.H_m, self.L_m = self.calc_consumption_quantities()
        self.initial_carbon_emissions = self.calc_total_emissions()
        self.utility = self.calc_utility()

        print("self.H_m",self.H_m)
        #print("self.L_m",self.L_m)
        #print("self.initial_carbon_emissions", self.initial_carbon_emissions)
        #print("self.utility", self.utility)

        return self.initial_carbon_emissions, self.utility
    
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
    "M": 3,
    "sector_substitutability": 5,
    "low_carbon_substitutability":np.asarray([10,10,2]),
    "prices_low_carbon": np.asarray([1,1,1]),
    "prices_high_carbon": np.asarray([0.9,0.9,0.9]),
    "carbon_price": 0,
}

low_carbon_preferences = np.asarray([0.5,0.5,0.5])
sector_preferences = np.asarray([0.2,0.2,0.6])
#budget = 1

H_mins = np.asarray([0,0,0])
#U_mins = np.asarray([1,1,0])#want 3rd to have no requirement
U_mins = np.asarray([0,0,0])#want 3rd to have no requirement
budget_list = np.linspace(1,20,100)
budget_med = np.median(budget_list)
#b_min = sum(H_mins*(params["prices_high_carbon"])) 
#print("b_min", b_min)
#b_min = 1
omega_m = np.asarray([0.3, 0.3, 0.4])#has to add up to 1 FOR COBB DOUGLASS

#test_subject = Individual_test(params,low_carbon_preferences,sector_preferences,H_mins)
test_subject_cobbs_min_u_prop_B = Individual_test(params,low_carbon_preferences,sector_preferences,H_mins, U_mins,budget_med, omega_m)

#carbon_tax_list = np.linspace(0,1,3)

data = []
data_u = []
for i in budget_list:
    #print("carbon price,budget",i,j)
    data_point, data_point_u= test_subject_cobbs_min_u_prop_B.calc_stuff_cobbs_min_u_prop_B(i)
    #data_point, data_point_u= test_subject.calc_stuff(i)
    data.append(data_point)
    data_u.append(data_point_u)


data_array = np.asarray(data)
data_u_array = np.asarray(data_u)
#print("data_array",data_array, data_array.shape)


fig, ax = plt.subplots()
plt.plot(budget_list, data_array)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Emissions flow")#row

fig, ax = plt.subplots()
plt.plot(budget_list, data_u_array)
ax.set_xlabel("Budget, B")#col
#ax.xaxis.set_label_position('top') 
ax.set_ylabel("Utility, U")#row


#multi_line_matrix_plot( data, budget_list, carbon_tax_list, get_cmap("plasma"), 0, "Budget", "Carbon price", "Emissions")
#multi_line_matrix_plot( data, budget_list, carbon_tax_list, get_cmap("plasma"), 1, "Budget", "Carbon price", "Emissions")

plt.show()

