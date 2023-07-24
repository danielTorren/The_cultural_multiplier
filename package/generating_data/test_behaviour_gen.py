import numpy as np
import matplotlib.pyplot as plt

def calc_a_b(mean,var):

    term_1 = ((mean*(1-mean)/var) - 1)
    print("term_1", term_1, mean, var)

    a = term_1*mean
    b = term_1*(1-mean)
    return a,b

def calc_idenities(behave):
    return np.mean(behave)

def generate_identity_and_behaviour_beta_beta(N,a_iden,b_iden, var_behave, M):

    indentities_beta = np.random.beta( a_iden, b_iden, size=N)

    #print("indentities_beta", indentities_beta)

    a_behave, b_behave = calc_a_b(indentities_beta,var_behave)

    #print("a_behave, b_behave", a_behave, b_behave)

    behave_a_b = zip(a_behave, b_behave)

    behaviours = [np.random.beta(a,b, size=M) for a,b in behave_a_b]

    identities = [np.mean(A) for A in  behaviours]

    return behaviours,identities, indentities_beta

def generate_identity_and_behaviour_beta_gaussian(N,a_iden,b_iden, var_behave, M):

    indentities_beta = np.random.beta( a_iden, b_iden, size=N)

    #print("indentities_beta", indentities_beta)

    #a_behave, b_behave = calc_a_b(indentities_beta,var_behave)

    #print("a_behave, b_behave", a_behave, b_behave)

    #behave_a_b = zip(a_behave, b_behave)

    behaviours_uncapped = np.asarray([np.random.normal(identity,var_behave, size=M) for identity in  indentities_beta])

    behaviours = np.clip(behaviours_uncapped, 0, 1)

    identities = [np.mean(A) for A in  behaviours]

    return behaviours,identities, indentities_beta


    #low_carbon_preference_list = [np.random.beta(self.a_low_carbon_preference, self.b_low_carbon_preference, size=self.M) for n in range(self.N)]
    #


np.random.seed(10)

N = 10
a_iden = 1
b_iden = 1
var_behave = 0.1
M = 3

#behaviours,identities, indentities_beta = generate_identity_and_behaviour_beta_beta(N,a_iden,b_iden, var_behave, M)
behaviours,identities, indentities_beta = generate_identity_and_behaviour_beta_gaussian(N,a_iden,b_iden, var_behave, M)

print("behaviours", behaviours)
print("identities", identities)
print("indentities_beta", indentities_beta)