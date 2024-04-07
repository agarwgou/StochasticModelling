import pandas
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib.pylab as plt
import datetime as dt
from scipy.optimize import least_squares

#Analytical Option Formulae

'''
Calculation of the value of the following European options:
- Vanilla call/put
- Digital cash-or-nothing call/put
- Digital asset-or-nothing call/put
based on the following models:
1. Black-Scholes model
2. Bachelier model
3. Black76 model
4. Displaced-diffusion model
'''
#------------------------------------------------------------------------------
#Black-Scholes Model

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def BlackScholesDCashCall(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(d2)

def BlackScholesDCashPut(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(-d2)

def BlackScholesDAssetCall(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1)

def BlackScholesDAssetPut(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    #d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(-d1)
#------------------------------------------------------------------------------
#Bachelier Model
    
def BachelierCall(S0, K,r, sigma, T):
    return np.exp(-r*T)*((S0-K)*norm.cdf((S0-K)/(sigma*np.sqrt(T)))+sigma*np.sqrt(T)*norm.pdf((S0-K)/(sigma*np.sqrt(T))))

def BachelierPut(S0, K,r, sigma, T):
    return np.exp(-r*T)*((K-S0)*norm.cdf((K-S0)/(sigma*np.sqrt(T)))+sigma*np.sqrt(T)*norm.pdf((K-S0)/(sigma*np.sqrt(T))))

def BachelierDCashCall(S0, K,r, sigma, T):
    return np.exp(-r*t)*norm.cdf((s0-k)/(sigma*np.sqrt(t)))

def BachelierDCashPut(S0, K,r, sigma, T):
    return np.exp(-r*t)*norm.cdf((k-s0)/(sigma*np.sqrt(t)))

def BachelierDAssetCall(S0, K,r, sigma, T):
    return np.exp(-r*T)*(S0*norm.cdf((S0-K)/(sigma*np.sqrt(T)))+sigma*np.sqrt(T)*norm.pdf((S0-K)/(sigma*np.sqrt(T))))

def BachelierDAssetPut(S0, K,r, sigma, T):
    return np.exp(-r*T)*(S0*norm.cdf((K-S0)/(sigma*np.sqrt(T)))-sigma*np.sqrt(T)*norm.pdf((K-S0)/(sigma*np.sqrt(T))))

#------------------------------------------------------------------------------
#Black76 Model

def Black76Call(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F*norm.cdf(c1) - K*norm.cdf(c2))

def Black76Put(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(K*norm.cdf(-c2) - F*norm.cdf(-c1))

def Black76DCashCall(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(c2)

def Black76DCashPut(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(-c2)

def Black76DAssetCall(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    #c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return F*disc*norm.cdf(c1)

def Black76DAssetPut(S, K, r, sigma, T):
    F = np.exp(r*T)*S
    c1 = (np.log(F/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    #c2 = c1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return F*disc*norm.cdf(-c1)
#------------------------------------------------------------------------------
#Displaced-Diffusion Model

def DisplacedDiffusionCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(c1) - ((1-beta)/beta*F + K)*norm.cdf(c2))

def DisplacedDiffusionPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(((1-beta)/beta*F + K)*norm.cdf(-c2) - F/beta*norm.cdf(-c1))

def DisplacedDiffusionDCashCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(c2)

def DisplacedDiffusionDCashPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*norm.cdf(-c2)

def DisplacedDiffusionDAssetCall(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(c1) - ((1-beta)/beta*F)*norm.cdf(c2))

def DisplacedDiffusionDAssetPut(S, K, r, sigma, T, beta):
    F = np.exp(r*T)*S
    c1 = (np.log(F/(F+beta*(K-F)))+(beta*sigma)**2/2*T) / (beta*sigma*np.sqrt(T))
    c2 = c1 - beta*sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F/beta*norm.cdf(-c1) - ((1-beta)/beta*F)*norm.cdf(-c2))

def BlackScholesDCashCall(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(d2)

def BlackScholesDCashCallDelta(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (np.exp(-r*T)*norm.pdf(d2))/(sigma*S*np.sqrt(T))

def BlackScholesDCashCallVega(S, K, r, sigma ,T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (-np.exp(-r*T)*d1*norm.pdf(d2))/(sigma)