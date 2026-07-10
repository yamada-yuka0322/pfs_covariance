from astropy.table import Table
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from astropy.constants import c

import numpy as np

import h5py

abacus_redshift = np.array([0.100  , 0.150 , 0.200  , 0.250 , 0.300  , 0.350 , 0.400  , 0.450 , 0.500  ,
       0.575, 0.650 , 0.725, 0.800  , 0.875, 0.950 , 1.025, 1.100  , 1.175,
       1.250 , 1.325, 1.400  , 1.475, 1.550 , 1.625, 1.700  , 1.850 , 2.000   ,
       2.250 , 2.500  ])

def load_lss(path):
    table = Table.read(path)
    selection = (table["Z"]>0.8)&(table["Z"]<1.6)
    return table[selection]

def load_random(path):
    table = Table.read(path)
    selection = (table["Z"]>0.8)&(table["Z"]<1.6)
    return table[selection]

def load_abacus_box(path):

    cat = CompaSOHaloCatalog(path, cleaned=True)  # , halo_lc=True)
    assert cat.halo_lc
    halo_pos = cat.halos['x_L2com']
    halo_vel = cat.halos['v_L2com']
    num_p = cat.halos['N']
    return np.array(halo_pos), np.array(halo_vel), np.array(num_p)

def load_abacus(path, np_min = 150):
    
    # fields to load
    fields_lc = [
        'pos_interp',
        'vel_interp',
        'redshift_interp',
        'N_interp',
    ]

    # load halo lc catalog
    subsamples = dict(A=True, rv=False)
    cat = CompaSOHaloCatalog(path, cleaned=True, subsamples=subsamples, fields=fields_lc)  # , halo_lc=True)
    assert cat.halo_lc
    halo_pos = cat.halos['pos_interp']
    halo_vel = cat.halos['vel_interp']
    redshift = cat.halos['redshift_interp']
    num_p = cat.halos['N_interp']
    selection = np.array(num_p) > np_min
    return np.array(halo_pos)[selection], np.array(halo_vel)[selection], np.array(redshift)[selection], np.array(num_p)[selection]

def load_lc(zmin, zmax, np_min = 150, radec = True):
    index_min = len(abacus_redshift[abacus_redshift < zmin])
    index_max = len(abacus_redshift[abacus_redshift < zmax]) + 1
    
    redshift_list = abacus_redshift[int(index_min):int(index_max):]
    
    pos = []
    vel = []
    redshift = []
    particle = []
    for z in redshift_list:
        path = f"/lustre/work/ryuichiro.hada/PFS/AbacusSummit/halo_light_cones/AbacusSummit_base_c000_ph000/z{z:.3f}/lc_halo_info.asdf"
        
        _pos, _vel, _redshift, _p = load_abacus(path)
        pos.append(_pos)
        vel.append(_vel)
        redshift.append(_redshift)
        particle.append(_p)
        
    pos = np.vstack(pos)
    vel = np.vstack(vel)
    redshift = np.concatenate(redshift).ravel()
    particle = np.concatenate(particle).ravel()
    
    selection = (redshift >= zmin) & (redshift <= zmax) & (particle >= np_min)
    pos = pos[selection]
    vel = vel[selection]
    redshift = redshift[selection]

    pos[:, 0] += 1000.0
    pos[:, 1] += 1000.0
    pos[:, 2] += 1000.0

    # line-of-sight unit vector
    los = pos / np.linalg.norm(pos, axis=1)[:, None]

    # radial peculiar velocity [km/s] if vel is in km/s
    v_los = np.sum(vel * los, axis=1)

    c_kms = c.to("km/s").value

    # observed redshift including LOS peculiar velocity
    z_obs = (1.0 + redshift) * (1.0 + v_los / c_kms) - 1.0

    if(radec):
        ra, dec, dist = xyz_to_radec(pos)

        return ra, dec, z_obs, redshift, dist

    return pos, vel, redshift, z_obs

def xyz_to_radec(pos):
    """
    Parameters
    ----------
    pos : (N,3) array
        Cartesian positions.

    Returns
    -------
    ra : degrees
    dec : degrees
    dist : same unit as input
    """

    # unit vectors of the new coordinate system
    ez = np.array([1., 1., 1.])
    ez /= np.linalg.norm(ez)

    ex = np.array([1., -1., 0.])
    ex /= np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    # coordinates in the rotated frame
    x = pos @ ex
    y = pos @ ey
    z = pos @ ez

    dist = np.sqrt(x**2 + y**2 + z**2)

    ra = np.degrees(np.arctan2(y, x))
    ra = (ra + 360.0) % 360.0

    dec = np.degrees(np.arcsin(z / dist))
    return ra, dec, dist

class Uchuu(object):
    
    def __init__(self, path):
        self.pos, self.vel, self.mass, self.id, self.upid, self.pid, self.rvir, self.rs = load_halos(path)
        
    
    def get_haloinfo(self, ID):
        return self.pos[self.id==ID], self.vel[self.id==ID], self.mass[self.id==ID], self.rvir[self.id==ID], self.rs[self.id==ID]
    
    def get_hosthalo(self, mass_min):
        host = (self.upid == -1) & (self.mass > mass_min)
        return self.pos[host], self.vel[host], self.mass[host], self.id[host], self.rvir[host], self.rs[host]
        
        
        
def load_halos(path):
    with h5py.File(path, 'r') as f:
        ID = np.array(f['id'][:])
        x = np.array(f['x'][:])
        y = np.array(f['y'][:])
        z = np.array(f['z'][:])
        
        vx = np.array(f['vx'][:])
        vy = np.array(f['vy'][:])
        vz = np.array(f['vz'][:])
        
        upid = np.array(f['upid'][:])
        pid = np.array(f['pid'][:])
        
        mass = np.array(f['Mvir'][:])
        
        rvir = np.array(f['Rvir'][:])
        rs = np.array(f['rs'][:])
        
    position = np.vstack([x,y,z]).T
    velocity = np.vstack([vx, vy, vz]).T
    
    return position, velocity, mass, ID, upid, pid, rvir, rs
    
    
    