import numpy as np
import pandas as pd

def simulate_Vasicek_One_Factor(r0: float = 0.1, a: float = 1.0, lam: float = 0.1, sigma: float = 0.2, T: int = 52, dt = 0.1) -> pd.DataFrame:
    """ Simulates a temporal series of interest rates using the One Factor Vasicek model
     interest_rate_simulation = simulate_Vasicek_One_Factor(r0, a, lam, sigma, T, dt)
    
     Args:
       r0 (float): starting interest rate of the vasicek process 
       a (float): speed of reversion" parameter that characterizes the velocity at which such trajectories will regroup around b in time
       lam (float): long term mean level that all future trajectories will evolve around  
       sigma (float): instantaneous volatility measures instant by instant the amplitude of randomness entering the system
       T (integer): end modelling time. From 0 to T the time series runs. 
       dt (float): increment of time that the process runs on. Ex. dt = 0.1 then the time series is 0, 0.1, 0.2,...
    
     Returns:
       N x 2 Pandas DataFrame where index is modelling time and values are a realisation of the underlying's price
       
     For more information see https://en.wikipedia.org/wiki/Vasicek_model
    """
    
    N = int(T / dt) + 1 # number of end-points of subintervals of length 1/dt between 0 and max modelling time T

    time, delta_t = np.linspace(0, T, num = N, retstep = True)

    r = np.ones(N) * r0

    for t in range(1,N):
        r[t] = r[t-1] * np.exp(-a*dt)+lam*(1-np.exp(-a*dt))+sigma*np.sqrt((1-np.exp(-2*a*dt))/(2*a))* np.random.normal(loc = 0,scale = 1)

    dict = {'Time' : time, 'Interest Rate' : r}

    interest_rate_simulation = pd.DataFrame.from_dict(data = dict)
    interest_rate_simulation.set_index('Time', inplace = True)

    return interest_rate_simulation