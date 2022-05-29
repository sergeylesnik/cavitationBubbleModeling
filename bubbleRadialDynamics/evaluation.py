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

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
# defaultdict will create a new list if the key is not found in the dictionary
from collections import defaultdict
import pickle


def cutInterval(dataDict, dtCut):
    '''Cut out specified number of initial periods to obtain only whole
    periods. Additionaly interpolate the first value to fit the start value
    of the integral.'''
    dataFirst = np.empty(0)
    t = dataDict['t']
    # .values() returns a reference: changes are applied directly to the dict
    data = list(dataDict.values())
    for i in range(len(data)):
        dataFirst = np.append(dataFirst, np.interp(dtCut, t, data[i][:]))
    # Find index where start value should be placed and insert corresp. values.
    tIndFirst = np.searchsorted(t, dtCut)
    t = np.insert(t, tIndFirst, dtCut)
    data = np.insert(data, tIndFirst, dataFirst, 1)
    # Delete arrays before start value.
    tCleared = np.extract(t >= dtCut, t)
    indToDel = range(0, len(t)-len(tCleared))
    data = np.delete(data, indToDel, 1)
    # Replace values in the dict with the interpolated ones.
    keys = list(dataDict.keys())
    for entry, valList in enumerate(dataDict):
        key = keys[entry]
        dataDict[key] = data[entry]


def calcDampingCoeffs(dat, tInt, nEval, par):
    t = dat['t']

    dVdt = 4*np.pi * dat['R']**2 * dat['dRdt']
    PiThMom = -dat['p_g']*dVdt
    PiTh = integrate.trapz(PiThMom, t)/tInt
    cumPiTh = integrate.cumtrapz(PiThMom, t)/tInt

    PiThQMom = -dat['dQdt']
    PiThQ = integrate.trapz(PiThQMom, t)/tInt
    cumPiThQ = integrate.cumtrapz(PiThQMom, t)/tInt

    Vs = (dat['Rs'])**3
    dVsdt = 3 * dat['Rs']**2 * dat['dRsdt']
    ps_g = dat['ps_g']
    PiThsMom = -(ps_g*dVsdt)
    ts = dat['ts']
    PiThs = integrate.trapz(PiThsMom, ts) / (2*np.pi*nEval)
    cumPiThs = integrate.cumtrapz(PiThsMom, ts)/(2*np.pi*nEval)

    PiThQsMom = -3*dat['dQsdt']
    PiThQs = integrate.trapz(PiThQsMom, ts) / (2*np.pi*nEval)
    cumPiThQs = integrate.cumtrapz(PiThQsMom, ts) / (2*np.pi*nEval)

    PiThs_surfTens = integrate.trapz(PiThsMom, ts) * \
        (1+2*par.sigma/(par.pstat*par.R0)) / (2*np.pi*nEval)

    PiViMom = dat['R'] * dat['dRdt']**2
    PiVi = 16*np.pi*par.mu * integrate.trapz(PiViMom, t)/tInt

    PiVisMom = dat['Rs'] * dat['dRsdt']**2
    PiVis = 6*par.omega*par.mu/(np.pi*par.pstat*nEval) * \
        integrate.trapz(PiVisMom, ts)

    V = 4/3*np.pi*dat['R']**3
    lnV = np.log(V)
    PiThlnMom = -(dat['p_g']*V)
    PiThln = integrate.trapz(PiThlnMom, lnV) / tInt
    PiThlnD = PiThln/par.PiNorm

    PiThVMom = -(dat['p_g'])
    PiThV = integrate.trapz(PiThVMom, V) / tInt
    PiThVD = PiThV/par.PiNorm

    PiThsVMom = -(dat['ps_g'])
    PiThsV = integrate.trapz(PiThsVMom, Vs) / (2*np.pi*nEval)

    lnVs = np.log(Vs)
    PiThslnMom = -(dat['ps_g']*Vs-1)
    PiThsln = integrate.trapz(PiThslnMom, lnVs) / (2*np.pi*nEval)

    # Save all variables from this function but internal and module 'par'
    calcDat = {k: v for k, v in locals().items() \
        if not (k.startswith('__') or k.startswith('par') or k == 'dat')}
    dat.update(calcDat)


def pickAndSaveDampingFactorsInCSV(struct, filePrefix, par):
    P_acSav = np.empty(0)
    PiVi = np.empty(0)
    PiTh = np.empty(0)
    for i, d in enumerate(struct):
        P_acSav = np.append(P_acSav, d['P_ac'])
        PiVi = np.append(PiVi, d['PiVi'])
        PiTh = np.append(PiTh, d['PiTh'])
    PiViD = PiVi/par.PiNorm
    PiThD = PiTh/par.PiNorm

    # Save damping functions
    dampingFile = filePrefix+'_PiTable.csv'
    P_acD = P_acSav/par.pstat
    dampingData = np.hstack((P_acD.reshape(-1, 1),
                             PiViD.reshape(-1, 1),
                             PiThD.reshape(-1, 1),
                             ))
    np.savetxt(dampingFile, dampingData, delimiter=',')


def filterModuleVars(storeDict, module, addDict={}):
    ''' Filter module to obtain paramter values only and exclude other internal
        classes. Optional: merge addDict into the storeDict'''
    for k in dir(module):
        v = getattr(module, k)
        if not k.startswith("__") and \
        (type(v) == float or type(v) == int or type(v) == np.float64):
            storeDict[k] = v

    # Merge additional dict into the storeDict
    storeDict.update(addDict)


def saveAsPickleFile(filePrefix, *dicts):
    parDatFile = filePrefix+'.pkl'
    with open(parDatFile, 'wb') as f:
        pickle.dump((dicts), f)


def dumpPhysAndCtrlParamsInPickle(paramDict, struct, filePrefix, parModule, c):
    ''' Dump struct and paramDict in binary format with compression.
        Readable only with python.'''

    # Filter par module and save result into paramDict
    filterModuleVars(paramDict, parModule, c)

    saveAsPickleFile(filePrefix, paramDict, struct)


def plot(d, struct, par):
    P_ac = np.empty(0)
    # Save damping factors in a dict with lists
    dampF = defaultdict(list)
    for i, dat in enumerate(struct):
        for k, v in dat.items():
            if (k.startswith('Pi') and not k.endswith('Mom')):
                dampF[k].append(v)
            if k == 'P_ac':
                P_ac = np.append(P_ac, v)
    PiViD = np.asarray(dampF['PiVi']) / par.PiNorm
    PiThD = np.asarray(dampF['PiTh']) / par.PiNorm
    t = d['t']

    fig_width = 8.27    # width in inches
    fig_height = 18.69  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'figure.figsize': fig_size}
    plt.rcParams.update(params)

    plt.figure()
    nplots = 9
    plt.subplot(nplots, 1, 1)
    plt.title('Toegel, $p_{{ \mathrm{{ac}} }} = {}\,$bar, '
                '$T = 20\,$C'.format(d['P_ac']/par.pstat))
    plt.plot(t / 1e-6, d['R'] / 1e-6, '-')
    plt.ylabel('$R$ [um]')

    plt.subplot(nplots, 1, 2)
    plt.plot(t / 1e-6, d['dRdt'], '-')
    plt.ylabel('$\dot R$ [m/s]')

    plt.subplot(nplots, 1, 3)
    plt.plot(t / 1e-6, d['n'] * par.avogadro, '-')
    plt.yscale('log')
    plt.ylabel('Anz. $H_2O$ Moleküle $N$')

    plt.subplot(nplots, 1, 4)
    plt.plot(t / 1e-6, d['T'], '-')
    plt.ylabel('$T$ [K]')
    plt.xlabel('$t$ [us]')

    plt.subplot(nplots, 1, 5)
    p_ext = par.pstat*(1 + d['ps_exc'])
    plt.plot(t / 1e-6, p_ext / 1e5, '-')
    plt.ylabel('$p_{\mathrm{ext}}$ [bar]')
    plt.xlabel('$t$ [us]')

    plt.subplot(nplots, 1, 6)
    plt.plot(t / 1e-6, d['p_g'] / 1e5, '-')
    plt.ylabel('$p_{\mathrm{b}}$ [bar]')
    plt.xlabel('$t$ [us]')

    plt.subplot(nplots, 1, 7)
    plt.plot(t / 1e-6, d['R_equ'] / 1e-6, '-')
    plt.ylabel('$R_{\mathrm{n}}$ [um]')

    plt.subplot(nplots, 1, 8)
    plt.plot(P_ac / par.pstat, PiViD, '-')
    plt.ylabel('$\Pi_{\mathrm{vi}}$ []')
    plt.plot(P_ac / par.pstat, PiThD, '-')
    plt.ylabel('$\Pi_{\mathrm{th}}$ []')
    plt.yscale('log')

    plt.tight_layout()
    #plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.5)
    plt.show()
    #plt.savefig('Toegel_20160314_Rmax500um_T20C.pdf')

    plt.figure()
    nplots = 3
    plt.subplot(nplots, 1, 1)
    plt.plot(d['Vs'], d['ps_g'])
    plt.ylabel('$p_g^* = p_g / p_0$')
    plt.xlabel('$V/V_0$')

    plt.subplot(nplots, 1, 2)
    lnVs = np.log(d['Vs'])
    ps_gVs = (d['ps_g']*d['Vs']-1)
    plt.plot(lnVs, ps_gVs)
    plt.ylabel('$p_g^* V^* - 1$')
    plt.xlabel('$ln(V^*)$')

    plt.subplot(nplots, 1, 3)
    plt.plot(t[0:-1], d['cumPiTh'])
    plt.ylabel('Cummulative Integration of $\Pi_{\mathrm{th}}$')
    plt.xlabel('Time t')

    print("")
