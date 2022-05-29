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
    Load saved pickle data and export characteristic maps (2D interpolation
    tables) in OpenFOAM format for an acoustic cavitation solver (meant to be
    placed in directory "constant" of the OpenFOAM case).
    The produced maps are responsible for the coupling between the acoustics,
    cavitation bubbles and liquid motion. These are based on the work of
    "Louisnard, O. (2012). A simple model of ultrasound propagation in a
    cavitating liquid. Part I: Theory, nonlinear attenuation and traveling wave
    generation. Ultrasonics Sonochemistry, 19(1), 56–65.
    https://doi.org/10.1016/j.ultsonch.2011.06.007"
"""

import numpy as np
import scipy.integrate as integrate
import pickle
from argparse import ArgumentParser
from pathlib import Path

import toegelModel as tm
import evaluation
import couplingToFoam as coupFoam


parser = ArgumentParser()
parser.add_argument("pklFile", metavar="file.pkl", type=str,
                    help="path to the pkl file to evaluate")
parser.add_argument("-o", "--outputFolder", metavar="outputFolder", type=str,
                    default="charactMapsForFOAM",
                    help="path to the folder for the results storage")
args = parser.parse_args()


tablePrefix = "/charactMap"

'''Phase shift is needed to match the formula from Louisnard 2012'''
phaseShift = np.pi * -1.0

with open(args.pklFile, 'rb') as f:
    par, d = pickle.load(f)

nPeriods = par['tInt']*par['freq']
P_ac = []
P_acBj = []
R0 = []
Ic = []
Is = []
t = []
VAv = []
RAv = []
PiVi = []
PiTh = []

# Loop over data sets for equilibrium radii
for iREqu, REquData in enumerate(d):
    print('Evaluating REqu =', REquData[0]['R0'])
    P_ac.append([])
    P_acBj.append([])
    Ic.append([])
    Is.append([])
    VAv.append([])
    RAv.append([])
    PiVi.append([])
    PiTh.append([])


    # Loop over pressure amplitudes
    for data in REquData:

        # Find the quantities in the dictionary
        for k, v in data.items():
            if (k == 'P_ac'):
                P_ac[iREqu].append(v)
            elif (k == 'R0') and (v not in R0):
                R0.append(v)
            elif (k == 't'):
                t = v
            elif (k == 'V'):
                V = v
            elif (k == 'R'):
                R = v
            elif (k == 'PiVi'):
                PiVi[iREqu].append(v)
            elif (k == 'PiTh'):
                PiTh[iREqu].append(v)

        IcCurr = integrate.trapz(V, np.cos(par['omega']*t + phaseShift)) \
            / (nPeriods*2*np.pi)
        IsCurr = integrate.trapz(V, np.sin(par['omega']*t + phaseShift)) \
            / (nPeriods*2*np.pi)
        VAvCurr = integrate.trapz(V, par['omega']*t) / (nPeriods*2*np.pi)
        RAvCurr = integrate.trapz(R, par['omega']*t) / (nPeriods*2*np.pi)

        Ic[iREqu].append(IcCurr)
        Is[iREqu].append(IsCurr)
        VAv[iREqu].append(VAvCurr)
        RAv[iREqu].append(RAvCurr)

    # OF doesn't have extrapolation function.
    # Thus, add (0, 0) entry to both tables Ic and Is.
    P_acBj[iREqu] = [0] + P_ac[iREqu]
    Ic[iREqu] = [0] + Ic[iREqu]
    Is[iREqu] = [0] + Is[iREqu]

# Save as OpenFOAM 2D interpolation table.
Path(args.outputFolder).mkdir(parents=True, exist_ok=True)

coupFoam.save_openfoam_charactMap(R0, P_acBj, Ic, 'Ic',
    args.outputFolder + tablePrefix + '_Ic_OF')
coupFoam.save_openfoam_charactMap(R0, P_acBj, Is, 'Is',
    args.outputFolder + tablePrefix + '_Is_OF')
coupFoam.save_openfoam_charactMap(R0, P_ac, VAv, 'VAv',
    args.outputFolder + tablePrefix + '_VAv_OF')
coupFoam.save_openfoam_charactMap(R0, P_ac, RAv, 'RAv',
    args.outputFolder + tablePrefix + '_RAv_OF')

# Smoothing and extrapolation of the damping functions if the Newton-Raphson
# solver in OpenFOAM struggles
#kSqrImSmooth = coupOF.smooth_upper_kSqrIm(P_ac, kSqrImNLD, 2*p0, 7)
#P_ac, kSqrImSmooth = coupOF.extrapolate_right_bound(P_ac, kSqrImSmooth, 10*p0)
coupFoam.save_openfoam_charactMap(R0, P_ac, PiVi, 'PiVi',
    args.outputFolder + tablePrefix + '_PiVi_OF')
coupFoam.save_openfoam_charactMap(R0, P_ac, PiTh, 'PiTh',
    args.outputFolder + tablePrefix + '_PiTh_OF')

# Save pkl file with the evaluation data
evaluation.saveAsPickleFile(args.outputFolder + tablePrefix,
    R0, P_ac, P_acBj, PiVi, Ic, Is, VAv, RAv)
