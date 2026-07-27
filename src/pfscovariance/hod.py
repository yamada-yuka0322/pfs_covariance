from halotools.empirical_models import HodModelFactory, OccupationComponent, NFWPhaseSpace
from halotools.sim_manager import UserSuppliedHaloCatalog

import math

import numpy as np

from scipy.stats import norm
from scipy.special import erf

class ModifiedHMQCentrals(OccupationComponent):
    def __init__(self, params):
        super().__init__(
            gal_type="centrals",
            threshold=-21,
            upper_occupation_bound=1,
        )
        self.param_dict = params

    def mean_occupation(self, **kwargs):
        halo_mass = np.asarray(
            kwargs["table"]["halo_mvir"],
            dtype=np.float64,
        )

        logMc = self.param_dict["logMc"]
        sigma_M = self.param_dict["sigma_M"]
        Ac = self.param_dict["Ac"]
        gamma = self.param_dict["gamma"]

        logM = np.log10(halo_mass)

        gaussian = (
            Ac
            / (np.sqrt(2.0 * np.pi) * sigma_M)
            * np.exp(
                -0.5
                * ((logM - logMc) / sigma_M) ** 2
            )
        )

        asymmetric_factor = (
            1.0
            + erf(
                gamma
                * (logM - logMc)
                / (np.sqrt(2.0) * sigma_M)
            )
        )

        mean_ncen = gaussian * asymmetric_factor

        # central occupationはBernoulli確率なので0〜1に制限
        return np.clip(mean_ncen, 0.0, 1.0)

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
        #logM1_ = self.param_dict['logM1_']
        alpha = self.param_dict['alpha']

        #logM1 = logM1_ + 1/alpha*np.log10(As)
        logM1 = 13.0
        dist = As*((halo_mass-10**logM0)/10**logM1)**alpha
        
        # Power-law model for satellite occupation
        return np.nan_to_num(np.maximum(dist, 0) * (halo_mass > 10**logM0))
    
def apply_strict_conformity(galaxies):
    """
    Remove satellites in halos with no centrals
    """
    galaxy_type = np.asarray(galaxies["gal_type"])

    central_mask = galaxy_type == "centrals"
    satellite_mask = galaxy_type == "satellites"

    central_host_ids = np.asarray(
        galaxies["halo_hostid"][central_mask]
    )

    satellite_host_ids = np.asarray(
        galaxies["halo_hostid"][satellite_mask]
    )

    satellite_has_central = np.isin(
        satellite_host_ids,
        central_host_ids,
    )

    keep = central_mask.copy()
    keep[satellite_mask] = satellite_has_central

    return galaxies[keep]
    
class Sat_Velocity(object):
    r"""Populate satellite galaxy velocity with gaussian distribution aroung the halo velocity.
    The dispersion is calculated as halo particle velocity dispersion times galaxy velocity bias fv.
    """
    def __init__(self, gal_type, fv):
        r"""
        Parameters
        ----------
        gal_type : string
            Type of galaxy

        fv : float
            galaxy velocity bias.
            
        Example
        --------
        >>> sat_velocity = Velocity(gal_type = 'satellites', fv = 1.23)

        """
        self.gal_type = gal_type
        self._mock_generation_calling_sequence = ['assign_velocity']
        self._galprop_dtypes_to_allocate = np.dtype([('vx', 'f8'),('vy', 'f8'), ('vz', 'f8')])
        self.list_of_haloprops_needed = ['halo_vx', 'halo_vy', 'halo_vz', 'halo_sigmav']
        self.fv = fv

    def assign_velocity(self, table, seed = 0):
        table['vx'][:] = table['halo_vx'][:]+norm.rvs(loc=0.0, scale=1.0, size=len(table["halo_vz"][:]))*self.fv/3**0.5*table['halo_sigmav'][:]
        table['vy'][:] = table['halo_vy'][:]+norm.rvs(loc=0.0, scale=1.0, size=len(table["halo_vz"][:]))*self.fv/3**0.5*table['halo_sigmav'][:]
        table['vz'][:] = table['halo_vz'][:]+norm.rvs(loc=0.0, scale=1.0, size=len(table["halo_vz"][:]))*self.fv/3**0.5*table['halo_sigmav'][:]
        
    
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
    rvir = table["rvir"] / 1000.0
    rs = table["rs"] / 1000.0
    
    sigmav = table["sigmav"]
    
    halocat = UserSuppliedHaloCatalog(redshift=redshift, Lbox=Lbox, particle_mass=particle_mass, 
                                 halo_x=x, halo_y=y, halo_z=z, halo_id=ids, halo_mvir=mass, halo_sigmav = sigmav,
                                 halo_upid = -np.ones(len(x)), halo_pid = -np.ones(len(x)), halo_hostid = -np.ones(len(x)),
                                halo_vx = vx, halo_vy = vy, halo_vz =vz, halo_rvir = rvir, halo_nfw_conc= rvir/rs)
    return halocat

    
def populate_galaxies(table, redshift, Lbox, particle_mass, params_cent, params_sat, np_min=50):
    num_halos = len(table)
    # Load a halo catalog
    # Create the HOD model
    custom_cens = CustomCentrals(params_cent)
    custom_sats = CustomSatellites(params_sat)
    sats_profile = sat_profile(redshift)
    sat_velocity = Sat_Velocity('satellites', params_sat['fv'])

    halocat = halo_cat(table, redshift, Lbox, particle_mass)
    
    custom_hod_model = HodModelFactory(centrals_occupation=custom_cens, satellites_occupation=custom_sats, satellites_profile=sats_profile, satellites_velocity=sat_velocity, redshift=redshift)
    # Populate the halo catalog with galaxies
    custom_hod_model.populate_mock(halocat, Num_ptcl_requirement=np_min)

    # Access the galaxy catalog
    galaxies = custom_hod_model.mock.galaxy_table
    #galaxies = apply_strict_conformity(galaxies)
    return galaxies