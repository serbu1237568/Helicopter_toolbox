import numpy as np
import sympy as sp
import math
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
def calculate_b1(p,alpha_d,a0,lambdai,):
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
def calculate_epsilon(p,lambda_i):
    epsilon0=lambda_i/p["mu"]
    return epsilon0
def calculate_B1_with_tail(p,a1,sA,hcd,wc,alpha_d,epsilon0,f=None,):
    if f==None:
        f=0
    alpha_t0_rad = math.radians(params["alpha_t0"])
    Cmf=0
    Cms=p["b"]*p["Mb"]*p["xg"]*p["e"]/(2*p["rho"]*sA*p["R"])
    kt=1+0.5*p["mu"]**2*p["Vt"]*p["at"]/(wc*p["h"]+Cms)
    B1=a1+(Cmf+hcd*p["h"]-wc*f-0.5*p["mu"]**2*p["Vt"]*p["at"]*(alpha_d+alpha_t0_rad-epsilon0))/kt*(wc*p["h"]+Cms)
    return B1
def calculate_A1(p,b1,Tt,wc,f=None):
    if f==None:
        f=0
    Cms=p["b"]*p["Mb"]*p["xg"]*p["e"]/(2*p["rho"]*sA*p["R"])
    A1=-b1-(wc*f+Tt/p["W"]*wc*p["ht"])/(wc*p["h"]+Cms)
    return A1
def calculate_phi(p,b1,A1,Tt):
    phi=-b1-A1-Tt/p["W"]
    return phi
def calculate_lambda_i_t(p,Tt):
    stAt=p["st"]*np.pi*(p["Rt"])**2
    ###supose same tip speed of main rotor and tail rotor sigmaR=sigmaRt
    tct=Tt/(0.5*p["rho"]*stAt*p["sigmaR"]**2)
    lambdait0=math.sqrt(p["st"]*tct/2)
    lambdait005=p["st"]*tct/2*p["mu"] 
    if p["mu"]==0:
        lambdait=lambdait0
    elif p["mu"]>0.05:
        lambdait=lambdait005 #### it has been considered that mu=mu_t
    elif 0<p["mu"]<0.05:
        lambdait = lambdait0 + (lambdait005 - lambdait0) * (p["mu"] / 0.05)
    return tct,lambdait
def calculate_theta_zero_t(p,tct,lambdait):
    theta0_t=3/(2*(1+3/2*p["mu"]**2))*(4/p["a"]*tct-lambdait)
    return theta0_t
def calculate_profile_power(p, sA):
    Pp=(p["delta"]/8) * p["rho"] * sA * p["sigmaR"]**3 * (1 + 3*p["mu"]**2)
    return Pp
def calculate_parasite_power(p):
    V = p["mu"] * p["sigmaR"]
    P_p=0.5 * p["rho"] * p["Sfp"] * V**3
    return P_p
def calculate_power_tail_rotor(p,lambdait,tct):
    stAt=p["st"]*np.pi*(p["Rt"])**2
    Pt=(p["delta"]/8*(1+3*p["mu"]**2)+lambdait*tct)*p["rho"]*stAt*p["sigmaR"]**3
    return Pt
def calculate_induced_power(p,vi0):
    Pi=(1+p["k"])*p["W"]*vi0
    return Pi
if __name__ == "__main__":
    #========================
    #Helicopter parameters
    #========================
    params={
        "W":45000,
        "s":0.05 ,  # solidity
        "R":8,
        "h":0.25,   ###for main rotor
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
        "gamma":6,
        "alpha_t0":12,  ###degree
        "Vt":0.1,       ####tail volume ration st*lt/s*A
        "at":3.5,
        "tt":11,         ###### secondary rotor tail arm (lt*R)    
        "ht":0.2,
        "st":0.1,        ###### solidity of tail rotor   
        "Rt":1.4,         ###### tail rotor radius
        "k":0.17

        
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
    B1=calculate_B1(params,a1,sA,hcd,wc,f=None)   ###f is c.g position
    theta=calculate_theta(B1,a1,f1,hcd,wc)
    #========================================
    #TAIL PLANE INFLUENCE
    #========================================
    epsilon0=calculate_epsilon(params,lambdai)
    B1_tail=calculate_B1_with_tail(params,a1,sA,hcd,wc,alpha_d,epsilon0,f=None,)
    
    #=========================================
    #Lateral control to trim
    #=========================================
    Tt=Q/params["tt"]
    A1=calculate_A1(params,b1,Tt,wc,f=None)
    phi=calculate_phi(params,b1,A1,Tt)
    [tct,lambdait]=calculate_lambda_i_t(params,Tt)
    theta0_t=calculate_theta_zero_t(params,tct,lambdait)
    Pi=calculate_induced_power(params,vi0)    #### Induced Power
    Pp=calculate_profile_power(params, sA)    #### profile power
    P_p=calculate_parasite_power(params)      #### parasite power
    Pt=calculate_power_tail_rotor(params,lambdait,tct)   #####tail rotor
    Ptotal = Pi + Pp + P_p + Pt
