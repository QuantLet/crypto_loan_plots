import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm, expon
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

def drawBBMax(b: np.array, h: np.array):
    u = np.random.uniform(low=0, high=1, size=b.shape)
    # Draw the maximum of Brownian bridge that connects 0 to level b from time 0 to time h
    # See Going to Extremes: Correcting Simulation Bias in Exotic Option Valuation by David R. Beaglehole, Philip H. Dybvig & Guofu Zhouhttps://www.tandfonline.com/doi/pdf/10.2469/faj.v53.n1.2057?casa_token=bdbUPlyH6yEAAAAA:Kq-5k8ZgSslD3GAjyBl9STvhPCRjIFi2Jmk0GzlkRU-CCI__MKbZIvvzG-mB-N0wPl6FTeOHn2I
    return (b+np.sqrt(b**2 - 2*h*np.log(1-u)) )/2

def drawBBMin(b: np.array, h: np.array):
    # Draw the minimum of Brownian bridge that connects 0 to level b from time 0 to time h
    # See Going to Extremes: Correcting Simulation Bias in Exotic Option Valuation by David R. Beaglehole, Philip H. Dybvig & Guofu Zhouhttps://www.tandfonline.com/doi/pdf/10.2469/faj.v53.n1.2057?casa_token=bdbUPlyH6yEAAAAA:Kq-5k8ZgSslD3GAjyBl9STvhPCRjIFi2Jmk0GzlkRU-CCI__MKbZIvvzG-mB-N0wPl6FTeOHn2I
    return -drawBBMax(-b, h)

def get_S_tilde(S, r, kappa, time):
    return time, S*np.exp( (-r-kappa)*time )

def simulate_kou_model_paths(seed, S0, mu, sigma, _lambda, p, eta1, eta2, T, M, N):
    np.random.seed(seed)
    
    # Time discretization
    dt = T / M
    time = np.linspace(0, T, M + 1)

    jump_compensation = (1-p)*eta2/(eta2+1) + p*eta1/(eta1-1) - 1

    Njumps = np.random.poisson(_lambda * T, N)

    times_with_jumps = []
    base_times_mask = []

    S = []
    S_t_minus = []
    S_BBmin = []

    for i in range(N):
        _Njumps = Njumps[i]                                    # There are _Njumps in path i
        jump_times = np.random.uniform(0, T, _Njumps)          # Allocate jump times uniformly
        _times_with_jumps = np.concatenate([jump_times, time]) # Insert the jump times into the base time array
        _base_times_mask = np.ones_like(_times_with_jumps)     # Record which times are jump times
        _base_times_mask[:_Njumps] = 0

        sorted_indices = np.argsort(_times_with_jumps)         # Get sorted indices based on _times_with_jumps
        _times_with_jumps = _times_with_jumps[sorted_indices]  # Sort _times_with_jumps
        _base_times_mask = _base_times_mask[sorted_indices]
        
        times_with_jumps.append(_times_with_jumps)
        base_times_mask.append(_base_times_mask)

        _dt_arr = np.diff(_times_with_jumps)
        _dt_arr = np.insert(_dt_arr, obj=[0], values=0)
        
        _dW = np.random.normal(0, 1, _Njumps+M+1)
        
        _jump_sizes = np.where(
                        np.random.rand(_Njumps) < p,
                        np.random.exponential(scale=1/eta1, size=_Njumps),  # Upward jumps
                        -np.random.exponential(scale=1/eta2, size=_Njumps))  # Downward jumps
        
        _jump_sizes = np.concatenate([_jump_sizes, np.zeros(M+1)])[sorted_indices]
        _Xt = (mu - 0.5 * sigma**2 - _lambda*jump_compensation)*_dt_arr + sigma*np.sqrt(_dt_arr)*_dW + _jump_sizes
        _S = S0*np.exp(np.cumsum(_Xt))
        _S_t_minus = _S/np.exp(_jump_sizes)
        
        S.append(_S)
        S_t_minus.append(_S_t_minus)
        
        # Brownian bridge
        _b_level = np.log( _S_t_minus[1:] / _S[:-1] ) / sigma
        _BBmin   = drawBBMin(_b_level, _dt_arr[1:])
        _S_BBmin = _S[:-1]*np.exp(sigma*_BBmin)
        _S_BBmin = np.insert(_S_BBmin, obj=[0], values=[S0])
        S_BBmin.append(_S_BBmin)
        
    return times_with_jumps, base_times_mask, S, S_t_minus, S_BBmin

def compute_payoffs(times_with_jumps, base_times_mask, S, S_t_minus, S_BBmin, K, H, LP, r, kappa):
    payoff = []
    time = []
    S_basetime = []
    liquidation_mask = []

    for i in range(len(times_with_jumps)):
        # Retrieve the path information: S_t_minus, S_t, sampled Brownian Bridge minimum, a time array with basetime and jump time, and base time mask
        _S_t_minus = S_t_minus[i]
        _S         = S[i]
        _S_BBmin   = S_BBmin[i]
        _times_with_jumps = times_with_jumps[i]
        _base_times_mask = base_times_mask[i]
        
        # Transform prices to the kappa + r discounted version for barrier breach checks
        _S_tilde_t_minus = get_S_tilde(_S_t_minus, r, kappa, _times_with_jumps)[1]
        _S_tilde         = get_S_tilde(_S, r, kappa, _times_with_jumps)[1]
        _S_tilde_BBmin   = get_S_tilde(_S_BBmin, r, kappa, _times_with_jumps)[1]
        
        # Initialise a liquidation mask for recording the time of liquidation, and the payoff without liquidation
        _liquidation_mask = np.ones_like(_S_t_minus)
        _payoff = np.clip(_S_tilde - K, 0, None) * np.exp((r+kappa)*_times_with_jumps)
        
        # Check liquidation
        _liquidation_by_cont_mask = np.logical_and(_S_tilde_t_minus < H, _S_tilde_BBmin < H)
        _liquidation_by_jump_mask = _S_tilde < H
        
        # Search for the first time of liquidation 
        _liquidation_by_cont_time = _liquidation_by_cont_mask.argmax()
        _liquidation_by_jump_time = _liquidation_by_jump_mask.argmax()

        if (_liquidation_by_cont_time == 0) and (_liquidation_by_jump_time == 0):
            # No liquidation
            j = np.inf
            
        elif (_liquidation_by_cont_time == 0) and (_liquidation_by_jump_time>0):
            # Liquidation by jump but there is no continuous movement causing liquidation
            j = _liquidation_by_jump_time
            _payoff[j: ] = np.clip(_S_tilde[j]-(1+LP)*K, 0, None)*np.exp((r+kappa)*_times_with_jumps[j])
            _liquidation_mask[j:] = 0
            
        elif (_liquidation_by_jump_time == 0) and (_liquidation_by_cont_time>0):
            # Liquidation by continuous movement but there is no jump causing liquidation
            j = _liquidation_by_cont_time
            _payoff[j: ] = np.clip(H-(1+LP)*K, 0, None)*np.exp((r+kappa)*_times_with_jumps[j])
            _liquidation_mask[j:] = 0
            
        else:
            # Liquidation in both continuous movement and jump, take the earliest liquidation time 
            if _liquidation_by_cont_time <= _liquidation_by_jump_time:
                j = _liquidation_by_cont_time
                _payoff[j: ] = np.clip(H-(1+LP)*K, 0, None)*np.exp((r+kappa)*_times_with_jumps[j])
                _liquidation_mask[j:] = 0

            else:
                j = _liquidation_by_jump_time
                _payoff[j: ] = np.clip(_S_tilde[j]-(1+LP)*K, 0, None)*np.exp((r+kappa)*_times_with_jumps[j])
                _liquidation_mask[j:] = 0
                
        time.append(_times_with_jumps[_base_times_mask == 1])
        S_basetime.append(_S[_base_times_mask == 1])
        payoff.append(_payoff[_base_times_mask == 1])
        liquidation_mask.append(_liquidation_mask[_base_times_mask == 1])
    
    time = np.array(time)
    S_basetime = np.array(S_basetime)
    payoff = np.array(payoff)
    liquidation_mask = np.array(liquidation_mask)
    
    return time, S_basetime, payoff, liquidation_mask

# Generate paths for the underlying asset using geometric Brownian motion
def LS_algo(S: np.array, payoffs: np.array, liquidation_mask: np.array, r: float, dt: float, T: float, regs: dict = None):

    if regs == None:
        # print('Learning optimal policy')
        regs = dict()
    else:
        # print('Pricing American payoff using policy given (in regs).')
        pass
    
    V = np.zeros_like(payoffs)
    V[:,-1] = payoffs[:,-1]
    
    # Step backward in time to apply the Longstaff-Schwartz algorithm
    for t in range(S.shape[1]-2, 0, -1):
        _S = S[:, t]
        _h = payoffs[:, t]
        _liq = liquidation_mask[:, t] # 1 for NOT liquidated, 0 for liquidated happened before or at the current time point
        
        # Discounted payoff at the next time step
        discounted_V = np.exp(-r * dt) * V[:, t+1]
        
        X = np.array([_S, _h, np.log(_S)]).T
        
        poly = PolynomialFeatures(degree=3)
        X = poly.fit_transform(X)/1000
        
        Y = discounted_V
        
        if regs.get(t) == None:
            reg = LinearRegression(fit_intercept=False).fit(X[_liq==1], Y[_liq==1])
            regs[t] = reg
        else:
            reg = regs[t]
            
        continuation_value = reg.predict(X)

        # continuation_value = np.clip(continuation_value, 0, None)

        # Exercise decision
        exercise_value = _h
        exercise = exercise_value > continuation_value
        # print(t, np.mean(exercise))

        # Update payoffs for paths where exercise is optimal
        V[:, t] = np.where(exercise, exercise_value, continuation_value)
        # V[:, t] = np.where(exercise, exercise_value, discounted_V)
        
        # Liquidated paths
        V[:, t] = np.where(_liq==0, discounted_V, V[:, t])

    _N = V.shape[0]
    # Discount the remaining payoff to time 0
    option_price = np.mean(np.exp(-r * dt) * V[:, 1])
    uq = option_price + np.std(payoffs)/np.sqrt(_N)*norm.ppf(0.95)
    lq = option_price + np.std(payoffs)/np.sqrt(_N)*norm.ppf(0.05)
    # print(f"American: {option_price:.5f} ({lq:.5f}, {uq:.5f})")
    
    EU_m = np.exp(-r * T)*np.mean(payoffs[:,-1])
    EU_uq = EU_m + np.std(np.exp(-r * T)*payoffs[:,-1])/np.sqrt(_N)*norm.ppf(0.95)
    EU_lq = EU_m + np.std(np.exp(-r * T)*payoffs[:,-1])/np.sqrt(_N)*norm.ppf(0.05)
    # print(f"European: {EU_m:.5f} ({EU_lq:.5f}, {EU_uq:.5f})")
    
    return regs, V
