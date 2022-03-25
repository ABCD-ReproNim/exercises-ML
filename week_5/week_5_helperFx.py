
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification 





class Create_MDD:
    
    def __init__(self, n_m, n_f):
        
        self.n_m = n_m
        self.n_f = n_f
        
        # Generate data for males
        m_x, m_y = make_classification(n_samples=self.n_m, n_features=20, n_informative=4, n_redundant=0, 
                                             n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep = 2)

        # Generate data for females
        f_x, f_y = make_classification(n_samples=self.n_f, n_features=20, n_informative=1, n_redundant=0, 
                                                 n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep = 1.5)

        # Skew female data
        f_x = f_x * 1.25

        # Stack and reshape
        m_x = np.hstack((m_x, np.zeros((n_m, 1)), m_y.reshape(m_y.shape[0], 1)))
        f_x = np.hstack((f_x, np.ones((n_f, 1)), f_y.reshape(f_y.shape[0], 1)))

        # Combine male and female
        X = np.concatenate((m_x, f_x))

        # Convert to dataframe
        df = pd.DataFrame(data = X)

        # Get column names
        with open('col_names.txt', 'r') as f:
            colnames = [nm.strip() for nm in f.readlines()]

        df.columns = colnames

        # Shuffle rows
        df = df.sample(frac = 1)

        # Get y and x
        self.y = df['depression']
        self.X = df.drop(columns = 'depression')
    

class Create_GAD:
    
    def __init__(self, n = 1000):
        
        # Num subs
        self.n = n
        
        # Create binary / cont version of GAD
        self.df = {'x': np.random.rand(n)}
        self.df['g_anxd'] = np.round(self.df['x'])
        
        # Indices of each case
        self.pos_inds = np.where(self.df['g_anxd'] == 1)[0]
        self.neg_inds = np.where(self.df['g_anxd'] == 0)[0]   
        
        # Create main outcome
        self.create_cont_var('income', 0, 1.5, 1.5, 1)
        
        # Create "random" vars
        self.create_cont_var('nihtbx_fluid', 1, 1, 2, 1)
        self.create_cont_var('nihtbx_flanker', 1, 2, 2, 1)
        self.create_cont_var('nihtbx_picvocab', 2, 2, 2, 1)
        self.create_cont_var('uppps_pos', 0, 2, 2, 4)
        self.create_cont_var('uppps_neg', 0, 1, 2, 2)
        
    def create_cont_var(self, var, mu1, mu2, sd1, sd2):
        
        "Create a continuous random variable broken down by outcome with given distribution params"
        
        self.df[var] = self.df['x'].copy()
        self.df[var][self.pos_inds] = np.random.normal(mu1, sd1, len(self.pos_inds))
        self.df[var][self.neg_inds] = np.random.normal(mu2, sd2, len(self.neg_inds))
        
    def create_bin_var(self, var, amt1, amt2):
        
        "Create a binary variable broken down by outcome"

        # Copy the values of the outcome and flip 
        self.df[var] = self.df['g_anxd'].copy()
        
        # Replace indices with randomly shuffled amt
        pos_rand_inds = np.random.choice(self.pos_inds, round(self.n * amt1), replace = False)
        neg_rand_inds = np.random.choice(self.neg_inds, round(self.n * amt2), replace = False)
        self.df[var][pos_rand_inds] = np.abs(self.df[var][pos_rand_inds] - 1)
        self.df[var][neg_rand_inds] = np.abs(self.df[var][neg_rand_inds] - 1)
    
    def add_context(self):
        
        "Hardcoded fx to include these variables"
        
        self.create_bin_var('access_to_care', .025, .125)
        self.create_cont_var('food_insecure', 2.2, -1.2, 1, 1.25)
        self.create_cont_var('neighb_violence', 228, 145, 20, 45)

        #self.create_bin_var('take_off', .017, .100)
        self.to_dframe()
        
    def to_dframe(self):
        
        "Convert to pandas dataframe"
        
        self.df = pd.DataFrame(self.df)
        

