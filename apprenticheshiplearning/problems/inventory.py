import numpy as np
from scipy.stats import poisson 
from apprenticheshiplearning.classes.mdp import Mdp

class Inventory:
    
    def __init__(self, max_inventory, demand_lambda, sell_price, holding_cost, order_cost, gamma, expert_occupancy_measure=None, c_estimate=None):
        self.max_inventory = max_inventory
        self.demand_lambda = demand_lambda
        self.sell_price = sell_price
        self.holding_cost = holding_cost
        self.order_cost = order_cost
        self.gamma = gamma
        self.expert_occupancy_measure = expert_occupancy_measure
        self.c_estimate = c_estimate

        # Connection with MDP class
        self.S = None
        self.A = None
        self.P = None
        self.v = None
        self.c = None
        self.mu_e = None
        self.c_hat = None

        # MDPs
        self.mdp_forward = None
        self.mdp_forward_dual = None
        self.mdp_IRL = None
        self.mdp_app = None

    def get_indexing(self, s, a):
        return s + len(self.S) * a
    
    def get_state_action(self, index):
        return index % len(self.S), index // len(self.S)

    def transitions(self, s1, s, a):
        # Total inventory after ordering
        T = s + a
        M = self.max_inventory

        # 1. Case: Stockout (s' = 0)
        if s1 == 0:
            # Stockout occurs if Demand D >= T
            # P(D >= T) is calculated using the Survival Function (1 - CDF)
            if T < 0:
                return 0.0 # Impossible state
            return poisson.sf(T - 1, self.demand_lambda)

        # 2. Case: Remaining Inventory is exactly the Capacity Cap (s' = M)
        # This state occurs if the resulting stock (T - D) is >= M, meaning Demand D <= T - M.
        elif s1 == M:
            demand_cap = T - M
            
            if demand_cap < 0:
                # If T < M, it is impossible for the stock to exceed M and be capped.
                # P(D <= negative) is 0.
                return 0.0
            
            # P(D <= T - M) is calculated using the Cumulative Distribution Function (CDF)
            # This correctly sums the probability mass for all low demands (D=0, D=1, ... up to T-M)
            # that would have resulted in stock > M.
            return poisson.cdf(demand_cap, self.demand_lambda)

        # 3. Case: Remaining Inventory is Below the Cap (1 <= s' < M)
        # This state only occurs if demand D is exactly T - s'.
        elif 1 <= s1 < M:
            demand_needed = T - s1
            
            # Check for impossible demand (s' > T)
            if demand_needed < 0:
                return 0.0
                
            # P(D = T - s') is calculated using the Probability Mass Function (PMF)
            # Note: If T > M, this implicitly excludes demands D <= T-M because those 
            # probability masses were accounted for in the s'=M case above.
            return poisson.pmf(demand_needed, self.demand_lambda)

        # 4. Case: Impossible states (s' < 0 or s' > M)
        else:
            return 0.0
        
    def cost(self, s, a):
        expected_income = np.sum([ self.sell_price * min(s + a, d) * poisson.pmf(d, self.demand_lambda) for d in range(self.max_inventory + 1)])
        ordering_cost = self.order_cost * a
        holding_cost = self.holding_cost * (s + a)
        return ordering_cost + holding_cost - expected_income
    
    def get_S(self):
        S = np.arange(self.max_inventory + 1)
        return S
    
    def get_A(self):
        A = np.arange(self.max_inventory + 1)
        return A
    
    def get_P(self, S, A):
        P = np.zeros((len(S), len(S), len(A)))
        for s1 in S:
            for a in A:
                for s in S:
                    P[s1, s, a] = self.transitions(s1, s, a)
        return P
    
    def get_v(self, S):
        v = np.zeros(len(S))
        for s in S:
            v[s] = 1/(len(S))
        return v
    
    def get_c(self, S, A):
        c = np.zeros((len(S) * len(A)))
        for s in S:
            for a in A:
                c[s + len(S) * a] = self.cost(s, a)

        #normalize the cost between -1 and 1
        c_min = np.min(c)
        c_max = np.max(c)

        print("Cost range: ", c_min, c_max)
        
        c = 2 * ((c - c_min) / (c_max - c_min)) - 1
        
        return c
    
    def get_mu_e(self):
        return self.expert_occupancy_measure
    
    def get_c_estimate(self):
        return self.c_estimate
    
    
    def get_mdp_forward(self):
        self.S = self.get_S()
        self.A = self.get_A()
        self.P = self.get_P(self.S, self.A)
        self.v = self.get_v(self.S)
        self.c = self.get_c(self.S, self.A)
        self.mu_e = self.get_mu_e()
        self.c_hat = self.get_c_estimate()
        self.mdp_forward = Mdp(self.S, self.A, self.P, self.v, self.c, self.gamma)