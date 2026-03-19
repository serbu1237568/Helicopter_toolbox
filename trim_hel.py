import numpy as np
import sympy as sp
####### TRIM HELICOPTER CALCULATOR  #######
####### NO TAIL EFFECT ####################
####### mu=fixed value ####################

def calculate_missing_values(p):
    sA=p["s"]*np.pi*(p["R"])**2
    d0=p["Sfp"]/sA
    wc=p["W"]/(p["rho"]*sA*p["sigmaR"]**2)   #considering tc=wc
    f1=0.5*p["mu"]**2*d0    #f1=0.5*d0*mu^2
    v0=np.sqrt(p["W"]/(2*p["rho"]*np.pi*p["R"]**2))
    return sA,d0,wc,f1,v0
def solve_vi0(p,v0):
    V=p["mu"]*p["sigmaR"]
    V_bar=V/v0
    vi=sp.symbols('vi')
    eq=(vi/v0)**4+V_bar**2*(vi/v0)**2-1
    solutions = sp.solve(eq, vi)
    return solutions
def choose_vi0(solution):
    real_positive=[]
    for sol in solution:
        sol_val = complex(sol.evalf())  # convert to numeric
        # Check if real (imaginary part ~ 0) and positive
        if abs(sol_val.imag) < 1e-10 and sol_val.real > 0:
            real_positive.append(sol_val.real)
    if not real_positive:
        return None  # no positive real solution found
    vi0=min(real_positive)
    return vi0

def calc_alpha_d(p,f1,wc,hcd=None):
    #first approximation of hcd
    if hcd is None:
        hcd=(1/4)*p["mu"]*p["delta"]
    alpha_d=-(f1+hcd)/wc
    return alpha_d
def calc_lamba_d(p,alpha_d,vi0,v0):
    lambdai=(vi0/v0)*v0/p["sigmaR"]
    lambdad=p["mu"]*alpha_d-lambdai
    return lambdad
def calculate_theta_zero(p,wc,lambdad):
    theta0=sp.symbols('theta0')
    eq=(p["a"]/4)*(2/3*theta0*(1-p["mu"]**2+9/4*p["mu"]**4)/(1+3/2*p["mu"]**2)+lambdad*(1-p["mu"]**2/2)/(1+3/2*p["mu"]**2))-wc
    solution=sp.solve(eq,theta0)
    return solution
def calculate_a1(p,theta0,lambdad):
    a1=(2*p["mu"]*(4/3*theta0+lambdad))/(1+3/2*p["mu"]**2)
    return a1
def calculate_hcd(p,lambdad,a1):
    hcd=1/4*p["mu"]*p["delta"]+p["a"]*lambdad/4*(0.5*a1-p["mu"]*theta0)
    return hcd
def calculate_a0(p,theta0,lambdad):
    a0=p["gamma"]/8*((theta0*1-19/18*p["mu"]**2+3/2*p["mu"]**2)/(1+3/2*p["mu"]**2)+4/3*lambdad*(1-p["mu"]**2/2)/(1+3/2*p["mu"]**2))
    return a0
def calculate_b1(p,alpha_d,a0,lambdai):
    ni=(1-np.sin(alpha_d))/(1+np.sin(alpha_d))
    b1=(4/3*(p["mu"]*a0+1.1*ni**0.5*lambdai))/(1+p["mu"]**2/2)
    return b1
def calculate_qc(p,lambdad,wc,hcd):
    qc=p["delta"]*(1+3*p["mu"]**2)/8-lambdad*wc-p["mu"]*hcd
    return qc
def calculate_B1(p,a1,sA,hcd,wc,f=None):
    #f is c.g. position
    if f==None:
        f=0
    Cmf=0
    Cms=p["b"]*p["Mb"]*p["xg"]*p["e"]/(2*p["rho"]*sA*p["R"])
    #Cms=(0.5*p["b"]*S*p["e"]*p["R"])/(p["rho"]*sA*p["omegaR"]**2*p["R"])
    #S=p["Mb"]*p["xg"]*p["R"]*(p["omegaR"]**2/p["R"]**2)
    B1=a1+(Cmf+hcd*p["h"]-wc*f)/(wc*p["h"]+Cms)
    return B1
def calculate_theta(B1,a1,f1,hcd,wc):
    #f1=0.5*d0*mu^2
    theta=B1-a1-hcd/wc-f1/wc
    return theta
if __name__ == "__main__":
    #========================
    #Helicopter parameters
    #========================
    params={
        "W":45000,
        "s":0.05 ,  # solidity
        "R":8,
        "h":0.25,
        "delta":0.013,
        "sigmaR":208,
        "Sfp":2.3,
        "b":4,    #number of blades
        "a":5.7,
        "Mb":74.7, #mass of one blade
        "xg":0.45,
        "e":0.04,
        "mu":0.3,   #tip speed ratio
        "rho":1.225,
        "gamma":6
    }
    sA,d0,wc,f1,v0=calculate_missing_values(params)
    solution=solve_vi0(params,v0)  ####solve equation for vi
    vi0=choose_vi0(solution)
    #=================================
    #FIRST EVALUATION VALUES
    #=================================

    alpha_d=calc_alpha_d(params,f1,wc)
    lambdad=calc_lamba_d(params,alpha_d,vi0,v0)
    theta0=calculate_theta_zero(params,wc,lambdad)
    theta0=float(theta0[0])
    a1=calculate_a1(params,theta0,lambdad)
    hcd=calculate_hcd(params,lambdad,a1)
    #===================================
    #Iterative solution
    #===================================
    # Iterative solution
    tolerance = 1e-6  # convergence criterion
    max_iter = 100     # prevent infinite loops
    error = 1.0        # initial error
    iteration = 0
    while error > tolerance and iteration < max_iter:
        iteration += 1
        # 1. Calculate alpha_d using current hcd
        alpha_d = calc_alpha_d(params, f1, wc, hcd)
        
        # 2. Calculate lambdad
        lambdad = calc_lamba_d(params, alpha_d, vi0, v0)
        
        # 3. Calculate theta0
        theta0_sol = calculate_theta_zero(params, wc, lambdad)
        theta0 = float(theta0_sol[0])  # take numeric value
        
        # 4. Calculate a1
        a1 = calculate_a1(params, theta0, lambdad)
        
        # 5. Calculate new hcd
        hcd_new = calculate_hcd(params, lambdad, a1)
        #6. Compute error as difference between consecutive hcd
        error = abs(hcd_new - hcd)
        # 7. Update hcd for next iteration
        hcd = hcd_new
    #======================================
    #OTHER PARAMETERS CALCULATION
    #======================================
    a0=calculate_a0(params,theta0,lambdad)
    lambdai=(vi0/v0)*v0/params["sigmaR"]
    b1=calculate_b1(params,alpha_d,a0,lambdai)
    qc=calculate_qc(params,lambdad,wc,hcd)
    Q=qc*params["rho"]*sA*params["sigmaR"]**2*params["R"] #torque
    power=params["W"]*vi0                                 #Power
    B1=calculate_B1(params,a1,sA,hcd,wc,f=None)   ###f is c.g position
    theta=calculate_theta(B1,a1,f1,hcd,wc)
