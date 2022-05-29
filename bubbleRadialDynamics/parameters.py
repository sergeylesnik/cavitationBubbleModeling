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
"""

import numpy as np
import scipy.constants as const

global k_B, avogadro, R_g, T0_Kelvin, theta_v

# Physical constants
k_B     = 1.38066e-23       # Boltzmann-Konstante [J / K]
avogadro= 6.02214e23        # Avogadro-Konstante [Teilchenzahl / mol]
R_g     = 8.31451           # universelle Gaskonstante [J / (mol * K)]

# Constant fraction factor for hard-core radius of argon/water mixture.
# Describes covolume for van-der-Waals equation of state. Toegel, Gompf 2000.
b_hcR = 8.86


class Argon:
    substance = 'Argon'
    M = 39.948  # [g / mol], Argon
    cp_ref = 0.52  # [J / (g*K)], Argon, cp at T0
    sigma = 3.432  # [Angstroem], Argon
    epsByK = 122.4  # [K], Argon
    # Table 13.2-3 Molecular theory of gases and liquids 1964.
    alphaPolar = 1.64  # [Angstroem**3], polarizability
    cpCoeff = 2.5  # heat capacity coefficient (cp/Rg) at const. pressure


class Air:
    substance = 'Air'
    M = 28.964  # [g / mol], Air
    cp_ref = 1.01  # [J / (g*K)], Air
    sigma = 3.617  # [Angstroem], Air
    epsByK = 97.0  # [K], Air
    alphaPolar_O2 = 1.6  # [Angstroem**3], polarizability
    alphaPolar_N2 = 1.76  # [Angstroem**3], polarizability
    # No literature found. Estimate.
    alphaPolar = 0.8*alphaPolar_N2 + 0.2*alphaPolar_O2
    nasa9_O2_TRange = \
        [   [200.00, 1000.00],
            [1000.00, 6000.00],
            [6000.00, 20000.00]     ]
    nasa9_O2_polyCoeffs = \
    [   [-3.425563420e+04,   4.847000970e+02,   1.119010961e+00,
            4.293889240e-03,  -6.836300520e-07,  -2.023372700e-09,
            1.039040018e-12,  -3.391454870e+03,   1.849699470e+01],
        [-1.037939022e+06,   2.344830282e+03,   1.819732036e+00,
            1.267847582e-03,  -2.188067988e-07,   2.053719572e-11,
            -8.193467050e-16,  -1.689010929e+04,   1.738716506e+01],
        [4.975294300e+08,  -2.866106874e+05,   6.690352250e+01,
           -6.169959020e-03,   3.016396027e-07,  -7.421416600e-12,
            7.278175770e-17,   2.293554027e+06,  -5.530621610e+02]   ]
    nasa9_N2_TRange = \
    [   [200.00, 1000.00],
        [1000.00, 6000.00],
        [6000.00, 20000.00]  ]
    nasa9_N2_polyCoeffs = \
    [   [2.210371497e+04,  -3.818461820e+02,   6.082738360e+00,
           -8.530914410e-03,   1.384646189e-05,  -9.625793620e-09,
            2.519705809e-12,   7.108460860e+02,  -1.076003744e+01],
        [5.877124060e+05,  -2.239249073e+03,   6.066949220e+00,
           -6.139685500e-04,   1.491806679e-07,  -1.923105485e-11,
            1.061954386e-15,   1.283210415e+04,  -1.586640027e+01],
        [8.310139160e+08,  -6.420733540e+05,   2.020264635e+02,
           -3.065092046e-02,   2.486903333e-06,  -9.705954110e-11,
            1.437538881e-15,   4.938707040e+06,  -1.672099740e+03]   ]


class Water:
    substance = 'Water'
    # Dampf in der Blase
    T0_Kelvin = 273.15  # Ice point [K]
    theta = [2295, 5255, 5400]  # Vibrational temperatures
    M = 18.016  # [g / mol], Water vapour
    cp_ref = 1.93  # [J / (g*K)], Water vapour

    # Force constants for Stockmeyer potential. Table 20, Molecular theory of
    # gases and liquids 1964. Used only for calculation of mass diffusion
    # coefficient, not for the viscosity.
    sigma_Stock = 2.65
    epsByK_Stock = 380
    dipole = 1.83  # [Debeye] Dipole moment


class O2:
    substance = 'Oxygen'
    M = 31.999  # [g / mol], Oxygen (O2)
    cp_ref = 0.919  # [J / (g*K)], Oxygen (O2)
    sigma = 3.433  # [Angstroem], Oxygen (O2)
    epsByK = 113.0  # [K], Oxygen (O2)
    alphaPolar = 1.6


class N2:
    substance = 'Nitrogen'
    M = 28.014
    sigma = 3.667
    epsByK = 99.8
    alphaPolar = 1.76  # [Angstroem**3], polarizability


class H2:
    substance = 'Hydrogen'
    M = 2.016  # [g / mol], Hydrogen (H2)
    cp_ref = 14.32  # [J / (g*K)], Hydrogen (H2)
    sigma = 2.915  # [Angstroem], Hydrogen (H2)
    epsByK = 38.0  # [K], Hydrogen (H2)
    alphaPolar = 0.79


class CO2:
    substance = 'Carbon Dioxide'
    M = 44.01  # [g / mol], Carbon Dioxide (CO2)
    cp_ref = 0.844  # [J / (g*K)], Carbon Dioxide (CO2)
    sigma = 3.996  # [Angstroem], Carbon Dioxide (CO2)
    epsByK = 190  # [K], Carbon Dioxide (CO2)

def print_props(substance):
    name = substance.substance
    for attr in dir(substance):
        if not attr.startswith('__'):
            print("%s.%s = %r" % (name, attr, getattr(substance, attr)))


def toegel2000Fig3():

    global pac, t_start, t_end, n_per
    global R_equ_Ar, freq, omega, T_l, pstat, rho0, mu, sigma, c0
    global t0, R0, v0, T0, p0, cutCoeff
    global ncg, vap

    '''Physical parameters'''
    R_equ_Ar= 4.5e-6            # Ruheradius mit ausschl. Ar [m]
    pac     = 1.2e5             # Schalldruck [Pa]
    freq    = 26.5e3            # freqenz [Hz]
    omega   = 2 * np.pi * freq # Kreisfreqenz [rad]
    T_l     = 300               # Wassertemperatur [Kelvin]

    pstat   = 1e5               # statischer Druck [Pa]
    rho0    = 998.21            # Dichte der Fluessigkeit [kg / m ^ 3]
    mu      = 1e-3              # Viskositaet der Fluessigkeit [Pa * s]
    sigma   = 0.0725            # Oberflaechenspannung [N / m]
    c0      = 1500.             # Schallgeschwindigkeit [m / s]

    # Cut-off coefficient for diffusion boundary layer. Toegel = pi, Louis = 5.
    cutCoeff = const.pi

    # Define substances.
    ncg = Argon()
    vap = Water()
    print_props(ncg)
    print_props(vap)

    derivedParams(vap, ncg)

    '''Initial conditions'''
    # R0, v0, T0, R_equ_Ar are given with Toegel_ode() call
    t0      = 1e-6              # Skalierung der Zeit [s]
    R0      = R_equ_Ar
    v0      = 1e-9
    T0      = T_l # in K
    p0      = pstat
    n_per   = 2
    t_start = 0e-6
    t_end   = n_per/freq


def toegel2000Fig4():

    global pac, R_equ_Ar, T_l, freq, R0, T0

    '''All parameters but pac, R0 and T0 are the same as Fig. 3.
    Load Fig. 3 params and define pac, T0. Empty frequency which is a variable
    parameter.'''
    toegel2000Fig3()

    pac = 1.3e5
    R_equ_Ar = 5e-6
    T_l = 293.15
    freq = []

    R0      = R_equ_Ar
    T0      = T_l # in K


def louisnard2012Fig1():

    global R_equ_Ar, freq, omega, T_l, pstat, rho0, mu, sigma, c0
    global t0, R0, v0, T0, p0, cutCoeff, PiNorm
    global vap, ncg

    R_equ_Ar= 3e-6         # Equilibrium bubble radius with air only [m]
    freq    = 20e3         # Frequency [Hz]
    T_l     = 300          # Surrounding liquid temperature [Kelvin]
    pstat   = 101300       # Pressure of the surrounding liquid [Pa]
    rho0    = 1000         # Density of the liquid [kg / m ^ 3]
    mu      = 1e-3         # Dynamic viscosity of the liquid [Pa * s]
    sigma   = 0.0725       # Surface tension water-air pair [N / m]
    c0      = 1500         # Speed of sound [m / s]
    # Cut-off coefficient for diffusion boundary layer.
    # Toegel = pi; Louis = 5.
    cutCoeff = 5
    omega   = 2 * np.pi * freq  # Angular frequency [rad]

    # Define substances.
    ncg = Air()  # Non-Condesable Gas
    vap = Water()  # Vapor

    derivedParams(vap, ncg)

    '''Initial conditions'''
    # R0, v0, T0, R_equ_Ar are given with Toegel_ode() call
    t0      = 1e-6              # Skalierung der Zeit [s]
    R0      = R_equ_Ar
    v0      = 1e-9
    T0      = T_l # in K
    p0      = pstat
    # scaling term for damping factors Pi
    PiNorm = pstat * 4/3*np.pi*R_equ_Ar**3 * omega


def louisnard2012Fig2():

    global R_equ_Ar, R0, PiNorm

    '''All parameters but R0 are the same as Fig. 1.
    Load Fig. 1 params and override R0'''
    louisnard2012Fig1()

    '''Physical parameters'''
    R_equ_Ar= 8e-6              # Equillibrium radius with Argon only [m]
    R0      = R_equ_Ar
    PiNorm = pstat * 4/3*np.pi*R_equ_Ar**3 * omega


def louisnard2012Fig3():

    global pac

    '''All parameters but pac are the same as Fig. 1.
    Load Fig. 1 params and define pac'''
    louisnard2012Fig1()

    '''Physical parameters'''
    pac = 1.5*pstat              # acoustic pressure [Pa]


# Setup for a 4cm cubic cuvette with an immersed sonotrode of 1cm diameter
def nowak2013Sono1cmCuv4cm():

    global freq, T_l, T0, omega, R_equ_Ar, PiNorm

    '''All parameters but freq are reused from Louisnard'''
    louisnard2012Fig1()

    '''Physical parameters'''
    freq = 17.3e3
    T_l = 293

    T0 = T_l
    omega   = 2*np.pi*freq
    PiNorm = pstat * 4/3*np.pi*R_equ_Ar**3 * omega


def derivedCtrlParams(c):

    global freq

    c['nCut'] = c['nPer'] - c['nEval'] # number of initial periods to cut out
    c['dtCut'] = c['nCut']/freq    # time interval to cut out
    c['tInt'] = c['nEval']/freq
    c['tStart'] = 0e-6
    c['tEnd'] = c['nPer']/freq   # end time for the solver

def derivedParams(vap, ncg):

    global sigma_AB, epsByK_AB, Omega_AB
    global lambda_vap, lambda_ncg, Phi_vap_ncg, Phi_ncg_vap, datalist

    sigma_AB, epsByK_AB = \
        combining_rule(vap.sigma_Stock, ncg.sigma, vap.epsByK_Stock, ncg.epsByK,
                       ncg.alphaPolar, vap.dipole)
    Omega_AB = omega_diff(T_l / epsByK_AB)

    mu_vap = viscosity_waterVap(T_l)
    lambda_vap = conductivity_pure(vap.cp_ref, vap.M, mu_vap)
    mu_ncg = viscosity(ncg.M, T_l, ncg.sigma, ncg.epsByK)
    lambda_ncg = conductivity_pure(ncg.cp_ref, ncg.M, mu_ncg)
    Phi_vap_ncg = Phi_coeff(vap.M, ncg.M, mu_vap, mu_ncg)
    Phi_ncg_vap = Phi_coeff(ncg.M, vap.M, mu_ncg, mu_vap)

    # D and alpha are calculated in ToegelModel because of concentration
    # dependency


def combining_rule(sigma_n, sigma_p, epsByK_n, epsByK_p, alpha_n=None,
                   mu_p=None):
    ''' Overloaded function.

    1st variant accepts pairs of two non-polar molecules - then
    no differentiation between 1st and 2nd substance (ignore _n, _p indices).

    2nd variant eq. (8.6-3 - 8.6-5) from Molecular Theory of gases and liquids
    1964 accepts also pairs of polar(_p) and non-polar(_n) molecules.
    Requires two additional arguments. Force constants of polar component must
    be for Stockmeyer potential.'''

    if alpha_n is None:
        sigma_np = (sigma_n + sigma_p) / 2.  # [Angstroem]
        epsByK_np = np.sqrt(epsByK_n * epsByK_p)  # [K]
    else:
        alphaRed_n = alpha_n / sigma_n**3  # both in Angstroem
        # Conversion to CGS units: debeye = 1e-18 Fr*cm; 1J = 1e7 erg;
        # Angstroem = 1e8 cm
        muRed_p = \
            mu_p*1e-18 / np.sqrt(epsByK_p*const.k*1e7*(sigma_p*1e-8)**3)
        ksi = 1 + 0.25 * alphaRed_n * muRed_p**2 * np.sqrt(epsByK_p/epsByK_n)
        sigma_np = 0.5 * (sigma_n + sigma_p) * ksi**(-1/6)
        epsByK_np = np.sqrt(epsByK_n*epsByK_p) * ksi
    return sigma_np, epsByK_np


def omega_diff(kTByeps):
    '''Collision integral depends on epsByK und T_l, which is const.
    Thus, compute only once at the beginning.
    Fit from Transport Phenomena (2002) eq. (E.2-2).'''
    return 1.06036 / kTByeps ** 0.15610 \
           + 0.19300 / np.exp(0.47635 * kTByeps) \
           + 1.03587 / np.exp(1.52996 * kTByeps) \
           + 1.76474 / np.exp(3.89411 * kTByeps)


def viscosity(M, T, sigma, epsByK):
    '''Viscosity acc. to Transport Phenomena (2002) eq. (1.4-14).
    The result is of pure substance meaning formula is for monatomic gases.
    Scale result by 1e2 to convert g/(cm*s) to g/(m*s)'''
    kTByeps = T / epsByK
    # Dimensionless collision integral for viscosity.
    omega_visc = 1.16145 / kTByeps ** 0.14874 \
        + 0.52487 / np.exp(0.77320 * kTByeps) \
        + 2.16178 / np.exp(2.43787 * kTByeps)
    return 2.6693e-5 * np.sqrt(M * T) / (sigma**2 * omega_visc) * 1e2


def viscosity_waterVap(T):
    '''Viscosity of pure water acc. to Sutherland model from Molecular Theory
    of gases and liquids 1964 eq. (8.4-7) and table 8.4-4.
    Scale result by 1e5 to convert to g/(m*s).'''
    if T >= 273 and T < 450:
        kS = 140.2
        S = 459.4
    elif T >= 450 and T < 600:
        kS = 235.8
        S = 1051
    elif T >= 600 and T < 700:
        kS = 244.4
        S = 1108
    else:
        raise ValueError('Value of temperature for the computation of water '
                         'viscosity is out of range.')
    return kS * (T)**0.5 / (1 + S/T) * 1e-5


def conductivity_pure(Cp, M, mu):
    '''Condutivity of a pure substance acc. to Transport Phenomena (2002)
    eq. (9.3-15). Eucken Model for both monatomic and polyatomic gases.'''
    return (Cp + 1.25 * R_g/M) * mu


def Phi_coeff(M_A, M_B, mu_A, mu_B):
    '''Parameter needed for calculation of thermal conductivity of a gas
    mixture. Semiempirical approach. Transport Phenomena (2002) eq. (1.4-16)'''
    return 1 / np.sqrt(8) * (1+M_A/M_B)**-0.5 \
        * (1 + (mu_A/mu_B)**0.5 * (M_B/M_A)**0.25)**2


class Scaling:
    def __init__(self):
        self.S0 = 4*np.pi*R0**2
        self.V0 = 4/3*np.pi*R0**3

        # No physical meaning, only for scaling.
        self.n0 = p0*self.V0 / (R_g*T0)

        self.scaleArray = [
            ['Acceleration', R0 * omega**2],
            ['AmountOfSubstance', self.n0],
            ['AngularVelocity', omega],
            ['Diffusion', R0**2*omega],
            ['HeatFlux', self.S0*p0*R0*omega],
            ['Length', R0],
            ['MolarConcentration', p0/(R_g*T0)],
            ['SpecificEnergy', R_g*T0],
            ['Pressure', p0],
            ['PressureChangeRate', p0*omega],
            ['SubstanceChangeRate', self.n0*omega],
            ['SurfaceTension', p0*R0],
            ['Temperature', T0],
            ['TemperatureChangeRate', T0*omega],
            ['ThermalConductivity', p0*R0**2*omega/T0],
            ['Time', 1/omega],
            ['Velocity', R0*omega],
            ['Volume', self.V0],
        ]
        self.dimensionlessNumbers = [
            ['Euler', p0 / (rho0 * R0**2 * omega**2)],
            ['Helmholtz', R0*omega/c0],
            ['Reynolds', rho0 * R0**2 * omega/mu],
            ['Weber', rho0 * R0**3 * omega**2 / sigma]
        ]

    def dimless_number(self, numberName):
        return next(numberDef[1] for i, numberDef
                    in enumerate(self.dimensionlessNumbers)
                    if numberDef[0] == numberName)

    def scale_factor(self, propName):
        return next(scaleFac[1] for i, scaleFac in enumerate(self.scaleArray)
                    if scaleFac[0] == propName)

    def scale(self, propName, value):
        factor = self.scale_factor(propName)
        return value / factor

    def descale(self, propName, scaledValue):
        factor = self.scale_factor(propName)
        return scaledValue * factor
