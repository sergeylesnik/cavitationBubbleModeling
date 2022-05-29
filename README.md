# Cavitation bubble modeling
A collection of python scripts for modeling of cavitation bubbles.

## Bubble radial dynamics
The equations are based on the Toegel model[^Toegel2000], which additionally to
the radius evolution equation consider heat and mass transport through the
interface between the bubble interior and the surrounding liquid. Only acoustic
excitation is considered. The obtained temporal data is used to build integrals
according to the Louisnard model[^Louisnard2012a], which consist of two main
categories. First, the damping functions describe the attenuation of the
acoustic field due to thermal and viscous losses during the bubble oscillations.
Second, the functions needed for the calculation of the primary Bjerknes force
responsible for the bubble motion due to acoustic waves.

### Coupling to OpenFOAM
The mentioned above and some additional functions are computed in dependence on
the equilibrium bubble radius and the acoustic pressure amplitude. The outcomes
are saved in form of the 2D interpolation tables readable by OpenFOAM software.
Corresponding solvers are provided in the dedicated repository
(acousticCavitationOpenFOAM). The material properties are available for two
cases:
- A cubic reactor with an emerged sonotrode of 1cm diameter from
Nowak[^Nowak2013]
- A cylindrical tank with an emerged sonotrode of 12cm diameter from
Louisnard[^Louisnard2012b]

[^Toegel2000]: Toegel, R., Gompf, B., Pecha, R., & Lohse, D. (2000). Does Water
Vapor Prevent Upscaling Sonoluminescence? Physical Review Letters, 85(15),
3165–3168. https://doi.org/10.1103/PhysRevLett.85.3165

[^Louisnard2012a]: Louisnard, O. (2012). A simple model of ultrasound
propagation in a cavitating liquid. Part I: Theory, nonlinear attenuation and
traveling wave generation. Ultrasonics Sonochemistry, 19(1), 56–65.
https://doi.org/10.1016/j.ultsonch.2011.06.007

[^Nowak2013]: Nowak, T. (2013). Untersuchung von akustischen Strömungen im kHz-
und GHz-Bereich [PhD thesis, Georg August University of Göttingen].

[^Louisnard2012b]: Louisnard, O. (2012). A simple model of ultrasound
propagation in a cavitating liquid. Part II: Primary Bjerknes force and bubble
structures. Ultrasonics Sonochemistry, 19(1), 66–76.
https://doi.org/10.1016/j.ultsonch.2011.06.008