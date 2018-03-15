import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import PolynomialFeatures,scale

import pymc3 as pm
#import seaborn as sns; sns.set()
from scipy import stats, optimize
from sklearn.datasets import load_diabetes
from sklearn.cross_validation import train_test_split
from theano import shared

#Importing dataset
df = pd.read_csv('forestfires.csv')
df.drop(['X','Y','month','day',],1,inplace=True)

#Inputs and Output
X = scale(np.array(df.drop(['area'],1)))
y = np.array(df.area.apply(lambda x: np.log(x+1)))

np.random.seed(9)

#confidence intervals for our parameters
#pairs-bootstrap technique
##
##def draw_bs_pairs_linreg(x, y):
##    """Perform pairs bootstrap for linear regression."""
##
##    # Set up array of indices to sample from: inds
##    inds = np.arange(len(x))
##
##    # Generate replicates
##    
##    bs_inds = np.random.choice(inds, size=len(inds))
##    bs_x, bs_y = x[bs_inds], y[bs_inds]
##
##    return bs_x, bs_y
##
##X1,y1=draw_bs_pairs_linreg(X, y)
##X2,y2=draw_bs_pairs_linreg(X, y)


#Split Data
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2, random_state=42)

#Preprocess data for Modeling
model_input= shared(X_tr)
model_output= shared(y_tr)

#confidence intervals for our parameters
#bootstrap technique


#Generate Model
linear_model = pm.Model()
with linear_model:
    # Priors for unknown model parameters    

    alpha = pm.Normal("alpha", mu=0,sd=1)
    betas = pm.Normal("betas", mu=0, sd=1, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", tau=1) # you could also try with a HalfCauchy that has longer/fatter tails
    # Expected value of outcome

    mu = alpha + pm.math.dot(betas, model_input.T)
    # Likelihood (sampling distribution of observations)


    y = pm.Normal("y", mu=mu, sd=sigma, observed=model_output)
    # Obtain starting values via Maximum A Posteriori Estimate



#infering parameters
with linear_model:
    inference=pm.ADVI()
    approx = pm.fit(n=100,more_replacements={
        model_input:pm.Minibatch(X_tr),model_output:pm.Minibatch(y_tr)})

##plt.plot(-inference.hist)
##plt.ylabel('ELBO')
##plt.xlabel('iteration')

#intrepreting parameters
trace= approx.sample(draws=5000)
print(pm.summary(trace))

pm.plots.traceplot(trace)
plt.show()

ppc = pm.sample_ppc(trace[100:],model=linear_model,samples=200)
fig2 = plt.figure()

for i in range(50):

    plt.scatter(X_tr[i] * np.ones(len(ppc['y'][:, i])), ppc['y'][:, i], color='b', s=10, alpha=0.1)

    plt.scatter(X_tr[i], Y_tr[i], color='r', s=50)
    
prediction
ppc = pm.sample_ppc(trace[100:],model=linear_model,samples=200)
pred = ppc['y'].mean(axis=0)
score =pm.r2_score(y_tr,pred)
print('CV Score: {}'.format(score))
