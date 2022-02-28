#! /usr/bin/env/ python3

# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# ! Author: Mr. E. Dinesh Kumar (IIT Madras)                   |
# |============================================================|
# |                                                            |
# | Script to calculate the dispersion of a single capillary   |
# | wave based on the initial-value problem for                |
# | two-fluid case with equal kinematic viscosity and          |
# | one-fluid case (setting density and viscosity of the       |
# | upper fluid to 0).                                         |
# ! Ref:                                                       |
# ! A. Prosperetti, Motion of two superposed viscous fluids    |
# ! Phys. Fluids 24, 1981, pp. 1217-1223;                      |
# ! https://doi.org/10.1063/1.863522                           |
# |============================================================|
# |                                                            |
# ! Inspired from Matlab Script by                             |
# |   Dr. Fabian Denner, f.denner09@imperial.ac.uk,            |
# |   Dept. Mechanical Engineering, Imperial College London    |
# ! Ref:                                                       |
# !    http://dx.doi.org/10.5281/zenodo.58232                  |
# !    F. Denner, Frequency dispersion of small-amplitude      |
# !    capillary waves in viscous fluids, Phys. Rev. E (2016). |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

import numpy as np
from scipy.special import erfc


def Capillary2D_Prosperetti(Rho_Top, \
                            Rho_Bottom, \
                            Viscosity, \
                            SurfaceTension, \
                            WaveLength, \
                            TimeStepsPerPeriod, \
                            Amplitude, \
                            NoCycles):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Input data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Density upper fluid [kg/m^3]
    rhoU = Rho_Top

    # Density lower fluid [kg/m^3]
    rhoL = Rho_Bottom

    # Surface tension [N/m]
    sigma = SurfaceTension

    # Wavenumber [1/m] (No.of Waves)
    k = 2. * np.pi / WaveLength

    # Time steps per period [s]
    # TimeStepsPerPeriod

    # Total Time steps

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Calculation of constants for a cosine wave
    #   with a0 = amplitude and no initial velocity (u0 = 0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Density ratio
    beta = rhoL * rhoU / ((rhoL + rhoU) ** 2)

    # Kinematic viscosity
    v = Viscosity

    # Square of inviscid angular freq.
    wSq = sigma * k ** 3 / (rhoL + rhoU)
    omega = np.sqrt(wSq)

    # Initial amplitude
    a0 = Amplitude

    # Initial velocity of center point
    u0 = 0.

    # Characteristic timescale
    tau = 2. * np.pi / omega

    # Time step (different per case!)
    dt = tau / TimeStepsPerPeriod

    # Dimensionless viscosity

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Calculation of roots z1, z2, z3 and z4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vk2 = v * k ** 2

    p1 = 1
    p2 = -4 * beta * np.sqrt(vk2)
    p3 = 2 * (1 - 6 * beta) * vk2
    p4 = 4 * (1 - 3 * beta) * (vk2) ** (3 / 2)
    p5 = (1 - 4 * beta) * vk2 ** 2 + wSq

    p = [p1, p2, p3, p4, p5]

    # Vector with the four roots z1, z2, z3 and z4
    z = np.roots(p)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Calculation of interface height h
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    t = np.arange(0, NoCycles * tau, dt)

    part0 = 4 * (1 - 4 * beta) * vk2 ** 2 / (8 * (1 - 4 * beta) * vk2 ** 2 + wSq) * a0 * erfc(np.sqrt(vk2 * t))

    wSq_a0 = wSq * a0
    sqrt_t = np.sqrt(t)
    sqr_z = z ** 2

    Z1 = (z[1] - z[0]) * (z[2] - z[0]) * (z[3] - z[0])
    part1 = z[0] / Z1 * (wSq_a0 / (sqr_z[0] - vk2) - u0) * np.exp((sqr_z[0] - vk2) * t) * erfc(z[0] * sqrt_t)

    Z2 = (z[0] - z[1]) * (z[2] - z[1]) * (z[3] - z[1])
    part2 = z[1] / Z2 * (wSq_a0 / (sqr_z[1] - vk2) - u0) * np.exp((sqr_z[1] - vk2) * t) * erfc(z[1] * sqrt_t)

    Z3 = (z[0] - z[2]) * (z[1] - z[2]) * (z[3] - z[2])
    part3 = z[2] / Z3 * (wSq_a0 / (sqr_z[2] - vk2) - u0) * np.exp((sqr_z[2] - vk2) * t) * erfc(z[2] * sqrt_t)

    Z4 = (z[0] - z[3]) * (z[1] - z[3]) * (z[2] - z[3])
    part4 = z[3] / Z4 * (wSq_a0 / (sqr_z[3] - vk2) - u0) * np.exp((sqr_z[3] - vk2) * t) * erfc(z[3] * sqrt_t)

    eta = np.real(part0 + part1 + part2 + part3 + part4)

    return omega, t, eta
