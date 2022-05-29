# -*- coding: utf-8 -*-
"""
License
    Copyright 2022 Sergey Lesnik

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Description
    Solver functions for bubble radial dynamics based on the Toegel model:
    "Toegel, R., Gompf, B., Pecha, R., & Lohse, D. (2000). Does Water Vapor
    Prevent Upscaling Sonoluminescence? Physical Review Letters, 85(15),
    3165–3168. https://doi.org/10.1103/PhysRevLett.85.3165"
"""

import numpy as np
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import parameters as par


def set_(_R_eq_ncg, _p_ac, _omega, _T_l, _pstat, _rho0, _mu, _sigma,
         _c0, _t0):

    global R_eq_ncg, p_ac, omega, T_l, pstat, rho0, mu, sigma, c0, \
        t0, datalist, cutCoeff, b_hcR
    global M_vap, M_ncg, cp_ncgMol, sigma_AB, epsByK_AB, Omega_AB
    global lambda_vap, lambda_ncg, Phi_vap_ncg, Phi_ncg_vap
    global R_g, pi

    R_eq_ncg = _R_eq_ncg
    p_ac = _p_ac
    omega = _omega
    T_l = _T_l
    pstat = _pstat
    rho0 = _rho0
    mu = _mu
    sigma = _sigma
    c0 = _c0
    t0 = _t0

    # Holds intermidiate variables of ODEs during the solver iterations
    datalist = []

    # Misc. parameters
    cutCoeff = par.cutCoeff
    b_hcR = par.b_hcR
    R_g = par.R_g
    pi = np.pi

    # Parameters for diff_coeff
    M_vap = par.vap.M
    M_ncg = par.ncg.M
    cp_ncgMol = par.ncg.cp_ref * M_ncg  # Convert J/(g*K) to J/(mol*K).
    sigma_AB = par.sigma_AB
    epsByK_AB = par.epsByK_AB
    Omega_AB = par.Omega_AB
    lambda_vap = par.lambda_vap
    lambda_ncg = par.lambda_ncg
    Phi_vap_ncg = par.Phi_vap_ncg
    Phi_ncg_vap = par.Phi_ncg_vap


def Toegel_ode(R0, dR0, T0, R_eq_ncg, t_start, t_end):
    """Toegel Modell
    Initialisierung und Aufruf des Loesers
    (jeder Zeitschritt einzeln)
    """
    # sc_* are scaled variables
    global cs_vapR, ns_ncg, c0s, omegas, sigmas, Ts_l, ps_ac, ps_inf, ps_sat
    global scaling, He, Eu, Re, We

    scaling = par.Scaling()

    # Ideal gas law for the saturation pressure
    p_sat = saturation_pressure(T_l)

    # Scaling.
    ts_start = scaling.scale('Time', t_start)
    ts_end = scaling.scale('Time', t_end)
    omegas = scaling.scale('AngularVelocity', omega)
    Rs0 = scaling.scale('Length', R0)
    dRs0 = scaling.scale('Velocity', dR0)
    Ts0 = scaling.scale('Temperature', T0)
    Ts_l = scaling.scale('Temperature', T_l)
    Rs_eq_ncg = scaling.scale('Length', R_eq_ncg)
    c0s = scaling.scale('Velocity', c0)
    ps_ac = scaling.scale('Pressure', p_ac)
    ps_inf = scaling.scale('Pressure', pstat)
    ps_sat = scaling.scale('Pressure', p_sat)
    sigmas = scaling.scale('SurfaceTension', sigma)
    cs_vapR = ps_sat

    # The gas equillibrium radius corresponds to a bubble containing the gas
    # only.
    ns_ncg = (ps_inf+2*sigmas/Rs_eq_ncg) * Rs_eq_ncg**3 / Ts_l
    Rs0_mix = equillibrium_radius(ns_ncg, Ts_l, ps_inf-ps_sat)
    ns0 = ps_sat * Rs0_mix**3 / Ts_l

    Eu = scaling.dimless_number('Euler')
    He = scaling.dimless_number('Helmholtz')
    Re = scaling.dimless_number('Reynolds')
    We = scaling.dimless_number('Weber')

    # Initialize and call the integrator. Tollerance settings are important
    # to achieve the needed accuracy. LSODA performs best.
    sol = solve_ivp(Toegel_equation,
                    (ts_start, ts_end),
                    [Rs0, dRs0, ns0, Ts0],
                    method='LSODA',
                    dense_output=False,
                    rtol=1e-12,
                    atol=1e-14,
                    vectorized=False)

    dataDict = Toegel_equation_reconstruct(sol.t, sol.y)

    dataDict.update(
            {
                't': scaling.descale('Time', dataDict['ts']),
                'R':  scaling.descale('Length', dataDict['Rs']),
                'dRdt':  scaling.descale('Velocity', dataDict['dRsdt']),
                'n':  scaling.descale('AmountOfSubstance', dataDict['ns']),
                'T':  scaling.descale('Temperature', dataDict['Ts']),
                'p_g':  scaling.descale('Pressure', dataDict['ps_g']),
                'R_equ':  scaling.descale('Length', dataDict['Rs_equ']),
                'dQdt':  scaling.descale('HeatFlux', dataDict['dQsdt']),
                'D':  scaling.descale('Diffusion', dataDict['Ds'])
            }
        )

    # Filter non-ODE variables
    dataDict = {k: v for k, v in dataDict.items()
                if np.size(v) == np.size(dataDict['t'])}

    return dataDict


def Toegel_equation(ts, xs):
    """Compute one integration step
    using the equations from Toegel et al., Phys. Rev. Lett. 85, 3165 (2000).
    """

    Rs = xs[0]
    dRsdt = xs[1]
    ns = xs[2]
    Ts = xs[3]

    Ss = 3 * Rs**2
    Vs = Rs**3

    T = scaling.descale('Temperature', Ts)
    if par.ncg.substance == 'Argon':
        # cp is temperature independent for Argon
        cpCoeff_ncg = 2.5
        cpCoeff_ncgT0 = 2.5
    elif par.ncg.substance == 'Air':
        cpCoeff_O2 = cp_coeff(T, par.ncg.nasa9_O2_TRange,
                              par.ncg.nasa9_O2_polyCoeffs)
        cpCoeff_N2 = cp_coeff(T, par.ncg.nasa9_N2_TRange,
                              par.ncg.nasa9_N2_polyCoeffs)
        cpCoeff_ncg = 0.8*cpCoeff_N2 + 0.2*cpCoeff_O2
        cpCoeff_ncgT0 = cp_ncgMol / R_g

    cVCoeff_ncg = cpCoeff_ncg - 1

    ''' Amount Of Substance '''
    cs_ncg = ns_ncg / Vs

    # Apply dimensions to cs_vapR and ns_ncg and convert to [mol/m^3].
    c_vapR = scaling.descale('MolarConcentration', cs_vapR)
    c_ncg = scaling.descale('MolarConcentration', cs_ncg)
    D = diff_coeff(Omega_AB, sigma_AB, c_ncg, c_vapR)
    Ds = scaling.scale('Diffusion', D)
    ls_m = np.min([np.sqrt(Ds*Rs / abs(dRsdt)),
                   Rs/cutCoeff])
    cs_vap = ns / Vs
    dnsdt = Ss*Ds*(cs_vapR-cs_vap)/ls_m

    ''' Heat flux '''
    cs_mix = cs_vapR + cs_ncg
    x_vapR = cs_vapR / cs_mix
    x_ncg = cs_ncg / cs_mix
    lambda_mix = conductivity_mix(x_vapR, x_ncg, lambda_vap, lambda_ncg,
                                  Phi_vap_ncg, Phi_ncg_vap)
    lambdas_mix = scaling.scale('ThermalConductivity', lambda_mix)

    # All properties at T0.
    alphas_mix = lambdas_mix / (4*cs_vapR + cpCoeff_ncgT0*cs_ncg)
    ls_th = np.min([np.sqrt(alphas_mix * Rs / abs(dRsdt)),
                    Rs / cutCoeff])
    dQsdt = Rs**2 * lambdas_mix * (Ts_l-Ts) / ls_th

    ''' Pressure '''
    # Equillibrium radius is variable because of vapor condensation and
    # evaporation. Find roots of a polynomial to find it.
    ns_mix = ns+ns_ncg
    Rs_equ = equillibrium_radius(ns_mix, Ts_l, ps_inf)
    Vs_hc = (Rs_equ/par.b_hcR)**3  # Hard-core volume.

    # Radius has to be greater than the hard-core radius.
    if Rs < Rs_equ/par.b_hcR:
        logging.warning("R < R_equ / b_hcR.")

    ps_g = ns_mix*Ts / (Vs-Vs_hc)

    ''' Temperature '''
    T = scaling.descale('Temperature', Ts)
    sum1, sum2 = thermo_sums(T)
    Cs_v = cVCoeff_ncg * ns_ncg + (3. + sum2) * ns
    hs_v = 4*Ts_l
    us_v = (3+sum1)*Ts
    dVsdt = Ss * dRsdt
    dTsdt = 1/Cs_v * (-ps_g*dVsdt + 3*dQsdt + dnsdt*(hs_v-us_v))

    ''' Bubble Wall Acceleration
        Excitation with the phase of -sin() to be consistent with Louisnard
        2012 I and use his unaltered formulas for primary Bjerknes force
        calculation (with phase shift -pi).'''
    ps_exc = - ps_ac * np.sin(omegas*ts)
    dpsdt = (dnsdt*Ts+ns_mix*dTsdt)/(Vs-Vs_hc) - ns_mix*Ts*dVsdt/(Vs-Vs_hc)**2

    # If dRdt and c0s are equal, division by zero
    if dRsdt == c0s:
        logging.warning("dRdt == c0")
        dRsdt = dRsdt * (1.+1e-6)

    ddRsdtdt = 1/((1-dRsdt*He)*Rs) * \
        (
            Eu * ((1+dRsdt*He)*(ps_g-ps_exc-ps_inf) + Rs*dpsdt*He)
            - 4/Re * dRsdt/Rs
            - 2/(We*Rs)
            - (1-dRsdt*He/3) * 1.5*dRsdt**2
        )

    datalist.append([ts, ps_exc, ps_g, Rs_equ, dQsdt, Ds])

    return [dRsdt, ddRsdtdt, dnsdt, dTsdt]


def Toegel_equation_vectorized(ts, xs):

    if xs.shape[1] > 1:
        print(xs.shape)

    d = Toegel_equation_reconstruct(ts, xs)

    return [d['dRsdt'], d['ddRsdtdt'], d['dnsdt'], d['dTsdt']]


def Toegel_equation_reconstruct(ts, xs):
    """Compute one integration step
    using the equations from Toegel et al., Phys. Rev. Lett. 85, 3165 (2000).
    """

    Rs = xs[0]
    dRsdt = xs[1]
    ns = xs[2]
    Ts = xs[3]

    Ss = 3 * Rs**2
    Vs = Rs**3

    T = scaling.descale('Temperature', Ts)
    if par.ncg.substance == 'Argon':
        # cp is temperature independent for Argon
        cpCoeff_ncg = 2.5
        cpCoeff_ncgT0 = 2.5
    elif par.ncg.substance == 'Air':
        cpCoeff_O2 = []
        cpCoeff_N2 = []
        for T_i in T:
            cpCoeff_O2.append( cp_coeff(T_i, par.ncg.nasa9_O2_TRange,
                                  par.ncg.nasa9_O2_polyCoeffs) )
            cpCoeff_N2.append( cp_coeff(T_i, par.ncg.nasa9_N2_TRange,
                                  par.ncg.nasa9_N2_polyCoeffs) )
        cpCoeff_ncg = 0.8*np.array(cpCoeff_N2) + 0.2*np.array(cpCoeff_O2)
        cpCoeff_ncgT0 = cp_ncgMol / R_g

    cVCoeff_ncg = cpCoeff_ncg - 1

    ''' Amount Of Substance '''
    cs_ncg = ns_ncg / Vs

    # Apply dimensions to cs_vapR and ns_ncg and convert to [mol/m^3].
    c_vapR = scaling.descale('MolarConcentration', cs_vapR)
    c_ncg = scaling.descale('MolarConcentration', cs_ncg)
    D = diff_coeff(Omega_AB, sigma_AB, c_ncg, c_vapR)
    Ds = scaling.scale('Diffusion', D)
    ls_m = np.min([np.sqrt(Ds*Rs / abs(dRsdt)),
                   Rs/cutCoeff])
    cs_vap = ns / Vs

    dnsdt = Ss*Ds*(cs_vapR-cs_vap)/ls_m

    ''' Heat flux '''
    cs_mix = cs_vapR + cs_ncg
    x_vapR = cs_vapR / cs_mix
    x_ncg = cs_ncg / cs_mix
    lambda_mix = conductivity_mix(x_vapR, x_ncg, lambda_vap, lambda_ncg,
                                  Phi_vap_ncg, Phi_ncg_vap)
    lambdas_mix = scaling.scale('ThermalConductivity', lambda_mix)

    # All properties at T0.
    alphas_mix = lambdas_mix / (4*cs_vapR + cpCoeff_ncgT0*cs_ncg)
    ls_th = np.minimum(np.sqrt(alphas_mix * Rs / abs(dRsdt)),
                   Rs / cutCoeff)
    dQsdt = Rs**2 * lambdas_mix * (Ts_l-Ts) / ls_th

    ''' Pressure '''
    # Equillibrium radius is variable because of vapor condensation and
    # evaporation. Find roots of a polynomial to find it.
    ns_mix = ns+ns_ncg

    Rs_equ = np.empty(0)
    for ns_mix_i in ns_mix:
        Rs_equ = np.append(Rs_equ, equillibrium_radius(ns_mix_i, Ts_l, ps_inf))
    Vs_hc = (Rs_equ/par.b_hcR)**3  # Hard-core volume.

    # Radius has to be greater than the hard-core radius.
    if any(np.less(Rs, Rs_equ/par.b_hcR)):
        logging.warning("R < R_equ / b_hcR.")

    ps_g = ns_mix*Ts / (Vs-Vs_hc)

    ''' Temperature '''
    T = scaling.descale('Temperature', Ts)
    sum1 = np.empty(0)
    sum2 = np.empty(0)
    for T_i in T:
        sum1Scalar, sum2Scalar = thermo_sums(T_i)
        sum1 = np.append(sum1, sum1Scalar)
        sum2 = np.append(sum2, sum2Scalar)
    Cs_v = cVCoeff_ncg * ns_ncg + (3. + sum2) * ns
    hs_v = 4*Ts_l
    us_v = (3+sum1)*Ts
    dVsdt = Ss * dRsdt
    dTsdt = 1/Cs_v * (-ps_g*dVsdt + 3*dQsdt + dnsdt*(hs_v-us_v))

    ''' Bubble Wall Acceleration
    Excitation with the phase of -sin() to be consistent with Louisnard
    2012 I and use his unaltered formulas for primary Bjerknes force
    calculation (with phase shift -pi). '''
    ps_exc = - ps_ac * np.sin(omegas*ts)
    dpsdt = (dnsdt*Ts+ns_mix*dTsdt)/(Vs-Vs_hc) - ns_mix*Ts*dVsdt/(Vs-Vs_hc)**2

    ddRsdtdt = 1/((1-dRsdt*He)*Rs) * \
        (
            Eu * ((1+dRsdt*He)*(ps_g-ps_exc-ps_inf) + Rs*dpsdt*He)
            - 4/Re * dRsdt/Rs
            - 2/(We*Rs)
            - (1-dRsdt*He/3) * 1.5*dRsdt**2
        )

    # Save all the variables but internal of the function in a dict and return
    return {k: v for k, v in locals().items() if not k.startswith('__')}


def Toegel_ode_no_scale(R0, dR0, T0, R_eq_ncg, t_start, t_end):
    """Toegel Modell
    Initialize and run the ODE solver with non-scaled variables.
    """
    # sc_* are scaled variables
    global c_vapR, n_ncg, c0, omega, sigma, T_l, p_ac, p_inf, p_sat
    global scaling

    # Anfangswert fuer n berechnen mit dem idealen Gasgesetz
    # n ist Stoffmenge in mol
    p_sat = saturation_pressure(T_l)
    p_inf = pstat
    c_vapR = p_sat / (R_g*T_l)
    print('p_sat = {}'.format(p_sat))
    print('c_vapR = {}'.format(c_vapR))

    n_ncg = (p_inf+2*sigma/R_eq_ncg) * 4/3*pi*R_eq_ncg**3 / (R_g*T_l)
    R0_mix = equillibrium_radius_no_scale(n_ncg, T_l, p_inf-p_sat)
    n0 = p_sat * 4/3*pi*R0_mix**3 / T_l

    print('n_ncg = {}'.format(n_ncg))
    # Initialize and call the integrator. Tollerance settings are important
    # to achieve the needed accuracy. LSODA performs best.
    sol = solve_ivp(Toegel_equation_no_scale,
                    (t_start, t_end),
                    [R0, dR0, n0, T0],
                    method='LSODA',
                    dense_output=False,
                    rtol=1e-8,
                    atol=1e-8,
                    vectorized=False)
    t = sol.t
    R = sol.y[0]
    dRdt = sol.y[1]
    n = sol.y[2]
    T = sol.y[3]

    # 'datalist' contains invalid timesteps from the ode solver iterations.
    data = pick_valid_timesteps(t, datalist)

    p_ext = p_inf + data[:, 1]
    p_g = data[:, 2]
    R_equ = data[:, 3]
    dQdt = data[:, 4]
    D = data[:, 5]

    return t, R, dRdt, n, T, p_ext, p_g, R_equ, dQdt, D


def Toegel_equation_no_scale(t, x):
    """Compute one integration step
    using the equations from Toegel et al., Phys. Rev. Lett. 85, 3165 (2000).
    """

    R = x[0]
    dRdt = x[1]
    n = x[2]
    T = x[3]

    S = 4*pi * R**2
    V = 4/3*pi * R**3
    dVdt = S * dRdt

    ''' Amount Of Substance '''
    c_ncg = n_ncg / V
    D = diff_coeff(Omega_AB, sigma_AB, c_ncg, c_vapR)
    l_m = np.min([np.sqrt(D*R / abs(dRdt)),
                  R/cutCoeff])
    c_vap = n / V
    dn = S*D*(c_vapR-c_vap)/l_m

    ''' Heat flux '''
    c_mix = c_vapR + c_ncg
    x_vapR = c_vapR / c_mix
    x_ncg = c_ncg / c_mix
    lambda_mix = conductivity_mix(x_vapR, x_ncg, lambda_vap, lambda_ncg,
                                  Phi_vap_ncg, Phi_ncg_vap)
    alpha_mix = lambda_mix / ((4*c_vapR+2.5*c_ncg) * R_g)
    l_th = np.min([np.sqrt(alpha_mix * R / abs(dRdt)),
                   R / cutCoeff])
    dQdt = S * lambda_mix * (T_l-T) / l_th

    ''' Pressure '''
    # Equillibrium radius is variable because of vapor condensation and
    # evaporation. Find roots of a polynomial to find it.
    n_mix = n + n_ncg
    R_equ = equillibrium_radius_no_scale(n_mix, T_l, p_inf)
    V_hc = 4/3*pi * (R_equ/b_hcR)**3  # Hard-core volume.

    # Radius has to be greater than the hard-core radius.
    if R < R_equ/b_hcR:
        logging.warning("R < R_equ / b_hcR.")

    p_g = n_mix*R_g*T / (V-V_hc)

    ''' Temperature '''
    sum1, sum2 = thermo_sums(T)
    C_v = (1.5 * n_ncg + (3. + sum2) * n) * R_g
    h_v = 4 * R_g * T_l
    u_v = (3+sum1) * R_g * T
    dT = 1/C_v * (-p_g*dVdt + dQdt + dn*(h_v-u_v))

    ''' Bubble Wall Acceleration
    Excitation with the phase of -sin() to be consistent with Louisnard
    2012 I and use his unaltered formulas for primary Bjerknes force
    calculation. (with phase shift -pi)'''
    p_exc = - p_ac * np.sin(omega*t)
    dp = ((dn*T+n_mix*dT)/(V-V_hc) - n_mix*T*dVdt/(V-V_hc)**2) * R_g

    # If dRdt and c0s are equal, division by zero
    if dRdt == c0:
        logging.warning("dRdt == c0")
        dRdt = dRdt * (1.+1e-6)

    ddR = 1/(R*(1-dRdt/c0)) * \
        (
            1/rho0 * \
                (
                    (1+dRdt/c0) * (p_g-p_exc-p_inf)
                    + R/c0 * dp
                    - 4*mu * dRdt/R
                    - 2*sigma/R
                )
            - 1.5*dRdt**2 * (1-dRdt/(3*c0))
         )

    datalist.append([t, p_exc, p_g, R_equ, dQdt, D])

    return [dRdt, ddR, dn, dT]


def saturation_pressure(T):
    """Vapour pressure of water as a function of the temperature
    Equation from
    W. Wagner und A. Prusz, J. Phys. Chem. Ref. Data 31, 387--535 (2002)
    Section 2.3.1
    Temperature scale: ITS-90
    """

    # Parameters
    pc = 22.064e6 # [Pa]
    Tc = 647.096 # [K]
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    theta = 1 - T / Tc

    # Compute vapour saturation pv as a function of the temperature T.
    pv = pc * np.exp(Tc / T * (a1 * theta \
                                   + a2 * theta ** 1.5 \
                                   + a3 * theta ** 3 \
                                   + a4 * theta ** 3.5 \
                                   + a5 * theta ** 4 \
                                   + a6 * theta ** 7.5))
    return pv


def diff_coeff(Omega_AB, sigma_AB, c_ncg, c_vapR):
    '''Diffusion coefficient from Bird, Stewart, Lightfoot:
    Transport Phenomena (2002) eq. (17.3-11)
    Supply c in mol/m^3 !
    '''
    return 2.2646e-5 * np.sqrt(T_l * (1 / M_vap + 1 / M_ncg)) \
        / (sigma_AB**2 * Omega_AB * (c_vapR + c_ncg)) * 1e2 # [m^2 / s]


def conductivity_mix(x_A, x_B, k_A, k_B, Phi_AB, Phi_BA):
    '''Conductivity of a gas mixture of two species acc. to Transport Phenomena
    (2002) eq. (9.3-17). x_i are the mole fractions, k_i are conductivities
    of pure substances.'''
    return x_A*k_A/(x_A + x_B*Phi_AB) + x_B*k_B/(x_A*Phi_BA + x_B)


def cp_coeff(T, TRange, polyCoeffs):
    if T >= 20000:
        # Limit T to 20000K as the highest value.
        T = 20000
    elif T < 200:
        logging.warning("cp_coeff: T = %f. Temperature out of range (NASA9 \
                        polynomials). Setting cp_coeff(T = 200K)", T)
        T = 200

    for i, T_interval in enumerate(TRange):
        if (T >= T_interval[0]) and (T <= T_interval[1]):
            a = polyCoeffs[i]

    return a[0]*T**-2 + a[1]*T**-1 + a[2] + a[3]*T + a[4]*T**2 + a[5]*T**3 \
        + a[6]*T**4


def thermo_sums(T):
    sumT1 = 0
    sumT2 = 0
    for th in par.vap.theta:
        thByT = th / T
        sumT1 += thByT / (np.exp(thByT)-1)
        sumT2 += thByT**2*np.exp(thByT) / (np.exp(thByT)-1)**2
    return sumT1, sumT2


def equillibrium_radius(ns, Ts, ps):
    def f(Rs_equ):
        # Relation between equillibrium Radius and amount of substance.
        return ns*Ts - Rs_equ**3*(ps+2*sigmas/Rs_equ)*(1-1/b_hcR**3)

    # Find a root. An interval must be provided.
    return brentq(f, 0.01, 1e3)


def equillibrium_radius_no_scale(n, T, p):
    def f(R_equ):
        # Relation between equillibrium Radius and amount of substance.
        return n*T*R_g \
            - 4*pi/3*R_equ**3 * (p+2*sigma/R_equ)*(1-1/b_hcR**3)

    # Find a root. An interval must be provided.
    return brentq(f, 0.1e-6, 1e-3)


def pick_valid_timesteps(_t, _datalist):
    '''Retrieve intermediate variables from the ode function by using global
    datalist. Delete times which are not present in the solver output.'''
    data = np.asarray(_datalist)
    tAll = data[:, 0]

    # Delete repetitions in tAll.
    tUnique, indicesToKeep = np.unique(tAll, return_index=True)
    data = data[indicesToKeep, :]
    indicesToDel = []
    mask = np.in1d(tUnique, _t)
    for i in range(0, len(tUnique)):
        if not mask[i]:
            indicesToDel.append(i)
    data = np.delete(data, indicesToDel, 0)
    tCleared = data[:, 0]

    # Delete greater time steps.
    tailToDel = range(len(_t), len(tCleared))
    data = np.delete(data, tailToDel, 0)

    # Verify that both time arrays are the same.
    if not np.array_equiv(_t, data[:, 0]):
        print("\n")
        logging.warning("pick_valid_timesteps(_t, _datalist): Time steps of \
                        retrieved intermidiate variables from \
                        the ODEs do not agree with the solver output.\n")
    return data


def create_dict(*args):
    return dict({i: eval(i) for i in args})
