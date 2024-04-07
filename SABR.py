#Declare Pricing functions required
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (8,6)

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err

def dd_calibration(dd_beta,strikes,vols,F,T,r,atm_vol):
    err=0.0
    for i,vol in enumerate(vols):
        err += (vol - dd_impliedVolatility(S,strikes[i],r,DisplacedDiffusion(F,strikes[i],r,atm_vol,T,dd_beta),T))**2
    
    return err

def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if (payoff.lower() == 'call'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalCall(S, K, r, x, T),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalPut(S, K, r, x, T),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        impliedVol = np.nan

    return impliedVol

def dd_impliedVolatility(S, K, r, price, T):
    if (K>S):
        impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalCall(S, K, r, x, T),
                                1e-12, 10.0)
        
    elif (K<S):
        impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalPut(S, K, r, x, T),
                                1e-12, 10.0)
    return impliedVol


def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def Black76Call(F,K,r,sigma,T):
    d1=(np.log(F/K) + 0.5*(sigma**2)*T) / (sigma * np.sqrt(T))
    d2=(np.log(F/K) - 0.5*(sigma**2)*T) / (sigma * np.sqrt(T))
    return np.exp(-r*T)* ( (F*norm.cdf(d1)) - (K*(norm.cdf(d2))))

def Black76Put(F,K,r,sigma,T):
    d1=(np.log(F/K) + 0.5*(sigma**2)*T) / (sigma * np.sqrt(T))
    d2=(np.log(F/K) - 0.5*(sigma**2)*T) / (sigma * np.sqrt(T))
    return np.exp(-r*T)* ( (K*norm.cdf(-d2)) - (F*(norm.cdf(-d1))))

def BachelierCall(S, K, r, sigma, T):
    sigma=sigma*S
    d = (S - K) / np.sqrt((sigma**2) * T )   
    C = np.exp(-r * T) * (((S - K) * norm.cdf(d)) + ((sigma * np.sqrt(T)) * norm.pdf(d)))
    return C

def BachelierPut(S, K, r, sigma, T):
    sigma=sigma*S
    d = (S - K) / np.sqrt((sigma**2) * T )   
    C = np.exp(-r * T) * (((K - S) * norm.cdf(-d)) + ((sigma * np.sqrt(T)) * norm.pdf(d)))
    return C

def DisplacedDiffusion(F,K,r,sigma,T,beta):
    if (K>S):
        return DisplacedDiffusionCall(F,K,r,sigma,T,beta)
    elif (K<S):
        return DisplacedDiffusionPut(F,K,r,sigma,T,beta)
    
def DisplacedDiffusionCall(F,K,r,sigma,T,beta):
    return Black76Call(F/beta,K+(((1-beta)/beta)*F), r , sigma * beta, T)

def DisplacedDiffusionPut(F,K,r,sigma,T,beta):
    return Black76Put(F/beta,K+(((1-beta)/beta)*F), r , sigma * beta, T)