from halotools.empirical_models import HodModelFactory, OccupationComponent, NFWPhaseSpace
from halotools.sim_manager import UserSuppliedHaloCatalog

import math

import numpy as np

class CustomCentrals(OccupationComponent):
    def __init__(self, params):
        # Call the parent constructor
        super().__init__(gal_type='centrals',threshold=-21, upper_occupation_bound=1)

        # Define custom parameters
        self.param_dict = params
        
    def mean_occupation(self, **kwargs):
        # Custom central occupation function (example)
        halo_mass = kwargs['table']['halo_mvir']
        logMc = self.param_dict['logMc']
        sigma_M = self.param_dict['sigma_M']
        Ac = self.param_dict['Ac']
        
        # Central occupation based on a Gaussian function
        return Ac/(2*math.pi)**0.5/sigma_M * np.exp(-(np.log10(halo_mass) - logMc)**2/2/sigma_M**2)
    
class CustomSatellites(OccupationComponent):
    def __init__(self, params):
        super().__init__(gal_type='satellites', threshold=-21, upper_occupation_bound=float('inf'))

        # Define custom parameters
        self.param_dict = params
        
    def mean_occupation(self, **kwargs):
        halo_mass = kwargs['table']['halo_mvir']
        As = self.param_dict['As']
        logM0 = self.param_dict['logM0']
        logM1_ = self.param_dict['logM1_']
        alpha = self.param_dict['alpha']

        logM1 = logM1_ + 1/alpha*np.log10(As)
        dist = As*((halo_mass-10**logM0)/10**logM1)**alpha
        
        # Power-law model for satellite occupation
        return np.nan_to_num(np.maximum(dist, 0) * (halo_mass > 10**logM0))
    
def sat_profile(redshift):
    return NFWPhaseSpace(
        mdef="vir",
        redshift=redshift,
        conc_mass_model="direct_from_halo_catalog",
        concentration_key="halo_nfw_conc",
        halo_boundary_key="halo_rvir"
    )

def halo_cat(table, redshift, Lbox, particle_mass):
    ids = table["ids"]
    
    x = table["pos"][:,0]
    y = table["pos"][:,1]
    z = table["pos"][:,2]
    
    vx = table["vel"][:,0]
    vy = table["vel"][:,1]
    vz = table["vel"][:,2]
    
    mass = table["mass"]
    rvir = table["rvir"]
    rs = table["rs"]
    
    halocat = UserSuppliedHaloCatalog(redshift=redshift, Lbox=Lbox, particle_mass=particle_mass, 
                                 halo_x=x, halo_y=y, halo_z=z, halo_id=ids, halo_mvir=mass, 
                                 halo_upid = -np.ones(len(x)), halo_pid = -np.ones(len(x)), halo_hostid = -np.ones(len(x)),
                                halo_vx = vx, halo_vy = vy, halo_vz =vz, halo_rvir = rvir, halo_nfw_conc= rvir/rs)
    return halocat


    
def populate_galaxies(table, redshift, Lbox, particle_mass, params_cent, params_sat):
    num_halos = len(table)
    # Load a halo catalog
    # Create the HOD model
    custom_cens = CustomCentrals(params_cent)
    custom_sats = CustomSatellites(params_sat)
    sats_profile = sat_profile(redshift)

    halocat = halo_cat(table, redshift, Lbox, particle_mass)
    
    custom_hod_model = HodModelFactory(centrals_occupation=custom_cens, satellites_occupation=custom_sats, satellites_profile=sats_profile, redshift=redshift)
    # Populate the halo catalog with galaxies
    custom_hod_model.populate_mock(halocat)

    # Access the galaxy catalog
    galaxies = custom_hod_model.mock.galaxy_table
    return galaxies

