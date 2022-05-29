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
    Run a solver for bubble radial dynamics based on the Toegel model:
    "Toegel, R., Gompf, B., Pecha, R., & Lohse, D. (2000). Does Water Vapor
    Prevent Upscaling Sonoluminescence? Physical Review Letters, 85(15),
    3165–3168. https://doi.org/10.1103/PhysRevLett.85.3165"

    The material properties are available for two cases:
    - A cubic reactor with an emerged sonotrode of 1cm diameter from
    "Nowak, T. (2013). Untersuchung von akustischen Strömungen im kHz- und
    GHz-Bereich [PhD thesis, Georg August University of Göttingen]."
    - A cylindrical tank with an emerged sonotrode of 12cm diameter from
    "Louisnard, O. (2012). A simple model of ultrasound propagation in a
    cavitating liquid. Part II: Primary Bjerknes force and bubble structures.
    Ultrasonics Sonochemistry, 19(1), 66–76.
    https://doi.org/10.1016/j.ultsonch.2011.06.008"

Acknowledgements
    This solver was inspired by "Bubble Models" from Hendrik Soehnholz
"""

import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import sys
from argparse import ArgumentParser
from pathlib import Path

import parameters as par
import toegelModel as tm
import evaluation


def main():
    parser = ArgumentParser()
    parser.add_argument("case", type=str,
                        choices=["Louisnard2012", "Nowak2013"],
                        help="The case whose parameters are to be used")
    parser.add_argument("-o", "--outputFolder", metavar="outputFolder",
                        type=str, default="",
                        help="path to the folder for the results storage")
    parser.add_argument("-n", "--numberOfThreads", type=int,
                        default=4,
                        help="path to the folder for the results storage")
    args = parser.parse_args()

    """Main Controls"""
    c = {}  # dict with control parameters
    c['nPer'] = 12.2  # number of time periods to solve; may be non-integer
    c['nEval'] = 2  # number of periods at the end to be evaluated, must be int

    # Parameters for the characteristic map
    c['Ps_acMin'] = 0.1  # normalized minimum acoustic pressure amplitude
    c['Ps_acMax'] = 3  # normalized maximum acoustic pressure amplitude
    c['dPs_ac'] = 0.05  # increment of normalized acoustic pressure amplitude
    c['R0Min'] = 1e-6  # minimum equilibrium radius in m
    c['R0Max'] = 25e-6  # maximum equilibrium radius in m
    c['dR0'] = 1e-6  # increment of equilibrium radius in m
    nProcs = args.numberOfThreads  # number of processors for the parallel run

    # File prefix for saving of results
    Path(args.outputFolder).mkdir(parents=True, exist_ok=True)
    filePrefix = args.outputFolder + "data_" + args.case

    print("\nLoad parameters according to " + args.case)
    print("========================")
    if (args.case == "Nowak2013"):
        par.nowak2013Sono1cmCuv4cm()
    elif (args.case == "Louisnard2012"):
        par.louisnard2012Fig1()
    else:
        raise ValueError("Provided case: " + args.case + " is not recognized")

    # Parameters for each ODE solver call
    par.derivedCtrlParams(c)
    P_acArr = np.arange(c['Ps_acMin'], c['Ps_acMax']+c['dPs_ac'], c['dPs_ac']) \
                     * par.pstat
    R0Arr = np.arange(c['R0Min'], c['R0Max']+c['dR0'], c['dR0'])
    c['nR0'] = len(R0Arr)
    c['nP_ac'] = len(P_acArr)

    # Structure to store all the results data including the time resolved
    # information
    struct = []

    for R0Curr in R0Arr:
        print('\nR0 = {0}'.format(R0Curr))

        # Update parameters for the current equilibrium radius
        par.R_equ_Ar = R0Curr
        par.R0 = R0Curr
        par.PiNorm = par.pstat * 4/3*np.pi*par.R_equ_Ar**3 * par.omega

        # Work around to be able to pass several arguments to the ODE function.
        # If pFunc is called, the arguments defined in partial will be passed
        # automatically.
        pFunc = partial(run_ODE_for_P_ac, c)

        time_start = time.time()

        # Parallel run. Pool.map() keeps order of runs acc. to the submitted
        # list
        with Pool(nProcs) as p:
            dat = p.map(pFunc, P_acArr.tolist())

        time_elapsed = time.time() - time_start
        print("  Solver wall clock time = {0:g} s".format(time_elapsed))

        # dat is a list with dict results returned by the ODE function
        struct.append(dat)

    # Get all the parameters from the corresp. module in a dict
    paramDict = {}
    evaluation.filterModuleVars(paramDict, par, c)

    # Delete the last seen R0 from the parameter dict to avoid confusion
    del paramDict['R0']

    # Save data
    evaluation.saveAsPickleFile(filePrefix, paramDict, struct)


def run_ODE_for_P_ac(c, P_ac):
    tm.set_(par.R_equ_Ar, P_ac, par.omega, par.T_l, par.pstat, par.rho0,
            par.mu, par.sigma, par.c0, par.t0)

    time_start = time.perf_counter()
    print("  Started computation for P_ac = {}".format(P_ac))
    sys.stdout.flush()

    # Arguments: R0, v0, T0, R_equ_Ar, t_start, t_end
    dat = tm.Toegel_ode(par.R0, par.v0, par.T0, par.R_equ_Ar, c['tStart'],
                        c['tEnd'])
    time_elapsed = time.perf_counter() - time_start
    print("  Finished computation for P_ac = {0};"
        " Thread time = {1:g} s".format(P_ac, time_elapsed))
    sys.stdout.flush()

    evaluation.cutInterval(dat, c['dtCut'])
    evaluation.calcDampingCoeffs(dat, c['tInt'], c['nEval'], par)

    # Save current characteristic map parameters in the data structure
    dat.update({'P_ac': P_ac})
    dat.update({'R0': par.R0})
    dat.update({'PiNorm': par.PiNorm})

    return dat


if (__name__ == "__main__"):
    main()
