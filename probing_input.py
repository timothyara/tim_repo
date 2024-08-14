#!/usr/bin/env python3
from pywarpx import picmi
import math
import numpy as np
from scipy.constants import c, e, m_e
# ----------
# Parameters
# ----------
use_cuda = True
# Order of the stencil for z derivatives in the Maxwell solver.
n_order = 32
Simulation_name = "UT^3_2d_WarpX"
# Import basic units
e0 = 8.854187817e-12
me = m_e
qe = e
# Laser profile
lambda_laser = 800.e-9
w_laser = 2 * math.pi * c / lambda_laser
# Plasma profile
n_e = 1.8e25  # The density in the labframe (electrons.meters^-3)
ne_plasma = n_e
w_plasma = np.sqrt((ne_plasma) * qe**2 / (e0 * me))
gfactor = np.sqrt(1 - (w_plasma**2 / w_laser**2))
lambda_plasma = (2 * np.pi * c * gfactor / w_plasma)
# The laser
a0 = 1.5  # Laser amplitude
tau_FWHM = 40e-15
tau = tau_FWHM * 0.8495
ctau = tau * c  # Laser duration
w0 = 6.e-6
# The longitudinal laser laser
a02 = 0.5
tau2_FWHM = 50.e-15
tau2 = tau2_FWHM * 0.8495
ctau2 = tau2 * c
w02 = 60e-6
# The simulation box
zmin = 0
zmax = round((1.9 * lambda_plasma + 2 * ctau), 7)
Nz_orig = int(40 * int((zmax - zmin) / (lambda_laser)))
Nz = int((Nz_orig //1024 + 1) * 1024)
extra_length_r = w0/2
# Transverse size
rmax = 4 * w0 + extra_length_r  # Length of the box along r (meters)
Nr = int(((4 * rmax / lambda_laser // 256) + 1) * 256)
# Domain decomposition
max_grid_size = 1024
blocking_factor = 256
# Ramp Shape
ramp_up = 100e-6
plateau = 3.e-3
ramp_down = 100e-6
blank = zmax
z0 = zmax - ctau  # Laser centroid
z02 = z0 - ctau/2
zfoc = blank+ ramp_up  # Focal position
Nm = 2  # Number of modes used
# The simulation timestep
dt = (zmax - zmin) / Nz / c  # Timestep (seconds)
# The moving window
v_window = c * gfactor  # Speed of the window
# The density profile
# The particles of the plasma
p_zmin = zmin  # Position of the beginning of the plasma (meters)
p_zmax = ramp_up + plateau + ramp_down + blank
p_rmax = rmax - extra_length_r # Maximal radial position of the plasma (meters)
blank_r=rmax-p_rmax
ramp_r=extra_length_r
plateau_r=(p_rmax-ramp_r)
####Calculate the density profile for tanh density
def calculate_a_b(blank, ramp_up):
    # Calculate the scaling factor
    alpha = np.arctanh(0.98)
    # Calculate b
    b = ramp_up / alpha
    # Calculate a
    a = blank + (alpha * b / 2) - b
    return a, b
# Parameters
a, b = calculate_a_b(blank, ramp_up)
print(blank, a, b)
p_nz = 2  # Number of particles per cell along z
p_nr = 2  # Number of particles per cell along r
p_nt = 4  # Number of particles per cell along theta
probe_focal_position=[0, 0, zmax]
probe_centroid_position=[0, 0, z02]
Nr = int(((4 * rmax / lambda_laser // 256) + 1) * 256)
blank_r=rmax-p_rmax
ramp_r=extra_length_r
plateau_r=(p_rmax-ramp_r)
##############################################################################
# WarpX conversions
# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e
# Number of time steps
max_steps = 15000   #int((p_zmax - p_zmin) / (gfactor * c * dt))
# Number of cells
nx = 2 * Nr
nz = Nz
# Physical domain
xmin = -rmax
xmax = rmax
zmin = zmin
zmax = zmax
# Create grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=['open', 'open'],
    upper_boundary_conditions=['open', 'open'],
    lower_boundary_conditions_particles=['absorbing', 'absorbing'],
    upper_boundary_conditions_particles=['absorbing', 'absorbing'],
    moving_window_velocity = [0., gfactor*c],
    warpx_max_grid_size=max_grid_size,
    warpx_blocking_factor=blocking_factor
)
# Particles: plasma electrons
plasma_density = 1.8e25
plasma_xmin = -p_rmax
plasma_ymin = None
plasma_zmin = blank
plasma_xmax = p_rmax
plasma_ymax = None
plasma_zmax = None
plasma_dist = picmi.AnalyticDistribution(
    density_expression=" ((1 + tanh(2 * (z - a) / b)) / 2) * ((abs(x) <= plateau_r) + (abs(x) > plateau_r)*(abs(x) <= p_rmax)*((p_rmax - abs(x)) / ramp_r) + (abs(x) > p_rmax)*0) * n_e",
    a=a,
    b=b,
    p_rmax=p_rmax,
    plateau_r=plateau_r,
    ramp_r=ramp_r,
    n_e=n_e,
    lower_bound=[plasma_xmin, plasma_ymin, plasma_zmin],
    upper_bound=[plasma_xmax, plasma_ymax, plasma_zmax],
    fill_in=True
)
electrons = picmi.Species(
    particle_type='electron',
    name='electrons',
    initial_distribution=plasma_dist
)
# Laser setup
laser = picmi.GaussianLaser(
    wavelength=lambda_laser,
    waist=w0,
    duration=tau,
    focal_position=[0, 0, zfoc],
    centroid_position=[0, 0, z0],
    propagation_direction=[0, 0, 1],
    polarization_direction=[0, 1, 0],
    a0=a0,
    fill_in=True
)
laser_antenna = picmi.LaserAntenna(
    position=[0., 0., z0],
    normal_vector=[0, 0, 1]
)
laser2 = picmi.GaussianLaser(
    wavelength=lambda_laser,
    waist=w02,
    duration=tau2,
    focal_position=[0, 0, zmax],
    centroid_position=[0., 0., z02],
    propagation_direction=[0, 0, 1],
    polarization_direction=[1, 0, 0],
    a0=a02,
    fill_in=True
)
laser_antenna2 = picmi.LaserAntenna(
    position=[0., 0., z02],
    normal_vector=[0, 0, 1]
)
# Electromagnetic solver
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method='Yee',
    cfl=1.,
    divE_cleaning=0
)
check = picmi.Checkpoint(
    period=10000,
    write_dir=".",
    warpx_file_prefix='Checkpoint'
)
# Diagnostics
diag_field_list = ['B', 'E', 'J', 'rho']
field_diag = picmi.FieldDiagnostic(
    name='diag1',
    grid=grid,
    period=1000,
    data_list=diag_field_list,
    write_dir='.',
    warpx_file_prefix='Python_FieldDiag',
    warpx_format='openpmd',
    warpx_openpmd_backend='h5'
)
# Set up simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape='cubic',
    warpx_use_filter=1,
    warpx_serialize_initial_conditions=1
)
# Add plasma electrons
plasma_layout = picmi.GriddedLayout(
    grid=grid,
    n_macroparticle_per_cell=[p_nz, p_nr, p_nt]
)
sim.add_species(
    electrons,
    layout=plasma_layout
)
# Add laser
sim.add_laser(
    laser,
    injection_method=laser_antenna
)
sim.add_laser(
    laser2,
    injection_method=laser_antenna2
)
# Add diagnostics
sim.add_diagnostic(field_diag)
#sim.add_diagnostic(check)
# Write input file
sim.write_input_file(file_name='input_script')
# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()
# Run simulation
run_python_simulation = True
if run_python_simulation:
    sim.step(max_steps)
else:
    sim.set_max_step(max_steps)
