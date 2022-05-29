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
from scipy import signal

def _tab(n=1):
    return "    "*n


def _writeHeader(dataName, fileName):
    with open(fileName, 'w') as f:
        f.write(
                "// A characteristic map for \n"
                "// " + dataName + "(REqu, pAc)\n"
                "// data(\n"
                "//     equilibrium radius,\n"
                "//     acoustic pressure amplitude\n"
                "// )\n"
                "//\n"
                "// Loaded via transportProperties.\n"
                "//\n"
                "// Table structure:\n"
                "//  (\n"
                "//      REqu 1\n"
                "//      (\n"
                "//          (pAc 1     " + dataName + " 1)\n"
                "//          (pAc 2     " + dataName + " 2)\n"
                "//          (...)\n"
                "//      )\n"
                "//      REqu 2\n"
                "//      (...\n"
                "//      )\n"
                "//  )\n"
                )


def save_openfoam_table(R0, P_ac, data, dataName, fileName):
    _writeHeader(dataName, fileName)

    with open(fileName, 'a') as f:
        f.write("(\n"
                "    (\n"
                "        {:.7g}\n".format(R0) +
                "        (\n"
                )
        for i, P in enumerate(P_ac):
            f.write(_tab(3) + "({:.6e}".format(P)
                    + _tab() + "{:.6e})\n".format(data[i]))
        f.write("        )\n"
                "    )\n"
                ");\n")


def save_openfoam_charactMap(R0Arr, P_acArr2D, data2DArr, dataName, fileName):
    _writeHeader(dataName, fileName)

    with open(fileName, 'a') as f:
        f.write("(\n")

        for i, R0ArrCurr in enumerate(R0Arr):
            f.write(_tab() + "(\n" +
                    _tab(2) + "{:.7g}\n".format(R0ArrCurr) +
                    _tab(2) + "(\n")

            for j, PCurr in enumerate(P_acArr2D[i]):
                f.write(_tab(3) + "({:.6e}".format(PCurr) +
                        _tab() + "{:.6e})\n".format(data2DArr[i][j]))
            f.write(_tab(2) + ")\n" +
                    _tab() + ")\n")
        f.write(");\n")


def smooth_upper_kSqrIm(x, y, xStart, aveBox):
    '''
    Smooth the kSqrIm curve for x > xStart, where bifurcation produces spikes.
    Smoothing the whole function would shift the steep rise around the Blake
    threshold.
    '''
    indices = [i for i, v in enumerate(x) if v > xStart]
    firstInexOfInsert = indices[0]
    ySmooth = signal.savgol_filter(y[indices], aveBox, 1)
    ySmooth = np.insert(ySmooth, 0, y[0:firstInexOfInsert])
    return ySmooth


def extrapolate_right_bound(x, y, xExtrap):
    '''
    Extrapolate linearly up to xExrap for better convergence of the Newton
    method.
    '''
    fit = np.polyfit(x[-3:-1], y[-3:-1], 1)
    straight_line = np.poly1d(fit)
    yExtrap = straight_line(xExtrap)
    x = np.append(x, xExtrap)
    y = np.append(y, yExtrap)
    return x, y


def compute_acSource_dpdn(omega, rho, U0):
    return rho * omega**2 * U0


def compute_acSource_displacement(dpdn, rho, omega):
    return dpdn / (rho * omega**2)


def convert_N_to_beta(N, R0):
    return N * 4/3*np.pi*R0**3


def convert_beta_to_N(beta, R0):
    return beta / (4/3*np.pi*R0**3)


if __name__ == "__main__":
    omega = 2*np.pi*20e+3
    rho = 1.0e+3
    U0 = 5e-6
    dpdn = compute_acSource_dpdn(omega, rho, U0)
    dpdn = 142e6
    U0 = compute_acSource_displacement(dpdn, rho, omega)