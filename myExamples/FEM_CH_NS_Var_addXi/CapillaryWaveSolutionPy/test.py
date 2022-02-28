#! /usr/bin/env/ python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   Example Usage:
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pylab as plt
from Capillary2D_Prosperetti import *

Rho_Top = 1.
Rho_Bottom = 1000.
Viscosity = 1. / 3000.
SurfaceTension = 0.01
WaveLength = 100
TimeStepsPerPeriod = 100
Amplitude = 0.01
NoCycles = 20

omega, t, eta = Capillary2D_Prosperetti(Rho_Top, \
                                        Rho_Bottom, \
                                        Viscosity, \
                                        SurfaceTension, \
                                        WaveLength, \
                                        TimeStepsPerPeriod, \
                                        Amplitude, \
                                        NoCycles)
#   normalise
t = t * omega
eta = eta / Amplitude

fig, ax = plt.subplots()

ax.plot(t, eta, '-', label=r'Prosperetti (1981); $\omega = $ ' + repr(omega))
ax.set_xlabel(r"$t^*$")
ax.set_ylabel(r"$\eta^*$")
plt.legend()
plt.show()
