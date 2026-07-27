from astropy.table import Table

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

from astropy.constants import c
import astropy.units as u
import astropy.cosmology.units as cu

import numpy as np

import h5py

import os

import asdf

abacus_redshift = np.array([0.100  , 0.150 , 0.200  , 0.250 , 0.300  , 0.350 , 0.400  , 0.450 , 0.500  ,
       0.575, 0.650 , 0.725, 0.800  , 0.875, 0.950 , 1.025, 1.100  , 1.175,
       1.250 , 1.325, 1.400  , 1.475, 1.550 , 1.625, 1.700  , 1.850 , 2.000   ,
       2.250 , 2.500  ])

def load_lss(path):
    table = Table.read(path)
    ROSETTE  = np.asarray(table["ROSETTE_R"], dtype=np.float64)
    selection = (table["Z"]>0.8)&(table["Z"]<1.6)&(ROSETTE > 0.2) & (ROSETTE < 1.5)
    return table[selection]

def load_random(path):
    table = Table.read(path)
    rand_ROSETTE  = np.asarray(table["ROSETTE_R"], dtype=np.float64)
    selection = (table["Z"]>0.8)&(table["Z"]<1.6)& (rand_ROSETTE > 0.2) & (rand_ROSETTE < 1.5)
    return table[selection]

def load_abacus_box(path, min_mass = 2e11):
    file = os.path.join(path, "halo_info_000.asdf")
    cat = CompaSOHaloCatalog(path, 
                             fields = ['v_L2com', 'x_L2com', 'N', 'id', 'r98_L2com', 'r25_L2com', 'sigmav3d_L2com'],
                             cleaned=True,
                             halo_lc=False)
    af = asdf.open(file)
    halo_pos = cat.halos['x_L2com'].data + np.float32(1000.0)
    x = halo_pos[:,0]
    y = halo_pos[:,1]
    z = halo_pos[:,2]
    
    halo_vel = cat.halos['v_L2com'].data
    num_p = cat.halos['N'].data
    
    mass = cat.halos['N'].data*af['header']['ParticleMassHMsun']
    ids = cat.halos['id'].data
    rvir = cat.halos['r98_L2com'].data * 1000.0
    rs = cat.halos['r25_L2com'].data * 1000.0
    sigmav = cat.halos['sigmav3d_L2com'].data

    upid = np.ones(len(ids))*(-1)
    pid = upid
    
    selection = (mass > min_mass)&(x<500)&(y<500)&(z<500)
    
    return halo_pos[selection], halo_vel[selection], mass[selection], ids[selection], rvir[selection], rs[selection], sigmav[selection], af['header']['ParticleMassHMsun']

def load_abacus(path, np_min = 150):
    
    # fields to load
    fields_lc = [
        'pos_interp',
        'vel_interp',
        'redshift_interp',
        'N_interp',
        'id',
        'r98_L2com',
        'r25_L2com',
        'sigmav3d_L2com'
    ]

    # load halo lc catalog
    subsamples = dict(A=True, rv=False)
    af = asdf.open(path)
    pm = af['header']['ParticleMassHMsun']
    cat = CompaSOHaloCatalog(path, cleaned=True, subsamples=subsamples, fields=fields_lc)  # , halo_lc=True)
    
    assert cat.halo_lc
    
    halo_pos = cat.halos['pos_interp'].data
    halo_vel = cat.halos['vel_interp']
    redshift = cat.halos['redshift_interp']
    num_p = cat.halos['N_interp']
    rvir = cat.halos['r98_L2com'].data
    rs = cat.halos['r25_L2com'].data
    sigmav = cat.halos['sigmav3d_L2com'].data
    
    selection = (np.array(num_p) > np_min)
    return np.array(halo_pos)[selection], np.array(halo_vel)[selection], np.array(redshift)[selection], np.array(num_p * pm)[selection], rvir[selection], rs[selection], sigmav[selection]

def load_lc(zmin, zmax, np_min = 150, radec = True):
    index_min = len(abacus_redshift[abacus_redshift < zmin])
    index_max = len(abacus_redshift[abacus_redshift < zmax]) + 1
    
    redshift_list = abacus_redshift[int(index_min):int(index_max):]
    
    pos = []
    vel = []
    redshift = []
    mass = []
    rvir = []
    rs = []
    sigmav = []
    for z in redshift_list:
        path = f"/lustre/work/ryuichiro.hada/PFS/AbacusSummit/halo_light_cones/AbacusSummit_base_c000_ph000/z{z:.3f}/lc_halo_info.asdf"
        
        _pos, _vel, _redshift, _p, _rvir, _rs, _sigmav = load_abacus(path, np_min)
        pos.append(_pos)
        vel.append(_vel)
        redshift.append(_redshift)
        mass.append(_p)
        rvir.append(_rvir)
        rs.append(_rs)
        sigmav.append(_sigmav)
        
    pos = np.vstack(pos)
    vel = np.vstack(vel)
    redshift = np.concatenate(redshift).ravel()
    mass = np.concatenate(mass).ravel()
    rvir = np.concatenate(rvir).ravel()
    rs = np.concatenate(rs).ravel()
    sigmav = np.concatenate(sigmav).ravel()
    
    
    selection = (redshift >= zmin) & (redshift <= zmax)
    pos = pos[selection]
    vel = vel[selection]
    redshift = redshift[selection]
    mass = mass[selection]
    rvir = rvir[selection]
    rs = rs[selection]
    sigmav = sigmav[selection]

    pos[:, 0] += 990.5
    pos[:, 1] += 990.5
    pos[:, 2] += 990.5

    # line-of-sight unit vector
    los = pos / np.linalg.norm(pos, axis=1)[:, None]

    # radial peculiar velocity [km/s] if vel is in km/s
    v_los = np.sum(vel * los, axis=1)

    c_kms = c.to("km/s").value

    # observed redshift including LOS peculiar velocity
    z_obs = (1.0 + redshift) * (1.0 + v_los / c_kms) - 1.0

    if(radec):
        ra, dec, dist = xyz_to_radec(pos)

        return ra, dec, z_obs, redshift, dist, mass, rvir, rs, sigmav

    return pos, vel, redshift, z_obs, mass, rvir, rs, sigmav

def galaxies_to_radec(
    galaxies,
    rsd=True,
    cosmo=None,
    observer=(0.0, 0.0, 0.0),
):
    if rsd and cosmo is None:
        raise ValueError("cosmo must be provided when rsd=True.")

    # Position: Mpc/h の数値として取り出す
    pos = np.column_stack([
        column_to_value(galaxies["x"]),
        column_to_value(galaxies["y"]),
        column_to_value(galaxies["z"]),
    ])

    # Velocity: km/s の数値として取り出す
    vel = np.column_stack([
        np.asarray(galaxies["vx"], dtype=float),
        np.asarray(galaxies["vy"], dtype=float),
        np.asarray(galaxies["vz"], dtype=float),
    ])

    observer = np.asarray(observer, dtype=float)
    pos = pos - observer

    # Rotated coordinate basis
    ez = np.array([1.0, 2.0, 1.0])
    ez /= np.linalg.norm(ez)

    ex = np.array([1.0, 0.0, -1.0])
    ex /= np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    x_rot = pos @ ex
    y_rot = pos @ ey
    z_rot = pos @ ez

    # Mpc/h
    dist = np.linalg.norm(pos, axis=1)

    if np.any(dist == 0):
        raise ValueError("A galaxy is located at the observer position.")

    ra = np.degrees(np.arctan2(y_rot, x_rot))
    ra = np.mod(ra, 360.0)

    dec = np.degrees(
        np.arcsin(np.clip(z_rot / dist, -1.0, 1.0))
    )

    if not rsd:
        return ra, dec, dist

    # --------------------------------------------------
    # Real-space distance -> cosmological redshift
    # --------------------------------------------------
    h = cosmo.h

    z_grid = np.linspace(0.0, 3.0, 10000)

    # Astropy: Mpc
    # simulation coordinates: Mpc/h
    distance_grid = cosmo.comoving_distance(z_grid).to_value(u.Mpc) * h

    if np.max(dist) > distance_grid[-1]:
        raise ValueError(
            f"Maximum galaxy distance is {np.max(dist):.1f} Mpc/h, "
            f"but the distance grid extends only to "
            f"{distance_grid[-1]:.1f} Mpc/h. Increase z_grid maximum."
        )

    if np.min(dist) < distance_grid[0]:
        raise ValueError(
            "Some galaxy distances are below the interpolation range."
        )

    z_cos = np.interp(dist, distance_grid, z_grid)

    # --------------------------------------------------
    # Line-of-sight peculiar velocity
    # --------------------------------------------------
    los_hat = pos / dist[:, None]

    # Positive means receding from observer
    v_los = np.sum(vel * los_hat, axis=1)  # km/s

    a = 1.0 / (1.0 + z_cos)

    # cosmo.H gives km/s/Mpc.
    # Multiplying by 1/h gives km/s/(Mpc/h).
    H_z = cosmo.H(z_cos).to_value(u.km / u.s / u.Mpc) / h

    # Mpc/h
    delta_dist = v_los / (a * H_z)
    observed_dist = dist + delta_dist

    c = 299792.458  # km/s

    # First-order peculiar velocity correction
    z_obs = z_cos + (1.0 + z_cos) * v_los / c

    return ra, dec, dist, observed_dist, z_obs


def column_to_value(column, unit=None):
    """
    Astropy Quantity / Column / ndarray を単位なし ndarray に変換する。
    """
    if hasattr(column, "to_value"):
        if unit is None:
            return np.asarray(column.value, dtype=float)
        return np.asarray(column.to_value(unit), dtype=float)

    if hasattr(column, "quantity"):
        quantity = column.quantity
        if unit is None:
            return np.asarray(quantity.value, dtype=float)
        return np.asarray(quantity.to_value(unit), dtype=float)

    return np.asarray(column, dtype=float)
    
    

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
    ez = np.array([1., 2., 1.])
    ez /= np.linalg.norm(ez)

    ex = np.array([1., 0., -1])
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

class Halo(object):
    def __init__(self, path, Type = "UCHUU"):
        self.halocat = None
        self.pm = None
        if(Type == "UCHUU"):
            pos, vel, mass, ID, upid, pid, rvir, rs = load_Uchuu(path)
        elif(Type == "MiniUCHUU"):
            pos, vel, mass, ID, upid, pid, rvir, rs = load_halos(path)
        elif(Type=='Abacus'):
            pos, vel, mass, ids, rvir, rs, sigmav, self.pm = load_abacus_box(path)
            self.halocat = Table({
                "pos": np.asarray(pos, dtype=np.float32),
                "vel": np.asarray(vel, dtype=np.float32),
                "ids": np.asarray(ids, dtype=np.float32),
                "mass": np.asarray(mass, dtype=np.float32),
                "rvir": np.asarray(rvir, dtype=np.float32),
                "rs": np.asarray(rs, dtype=np.float32),
                "sigmav": np.asarray(sigmav, dtype=np.float32)
            })

class Uchuu(object):
    
    def __init__(self, path, Type = "UCHUU"):
        if(Type == "UCHUU"):
            self.pos, self.vel, self.mass, self.id, self.upid, self.pid, self.rvir, self.rs, self.sigmav = load_Uchuu(path)
        elif(Type == "MiniUCHUU"):
            self.pos, self.vel, self.mass, self.id, self.upid, self.pid, self.rvir, self.rs = load_halos(path)
        
    
    def get_haloinfo(self, ID):
        return self.pos[self.id==ID], self.vel[self.id==ID], self.mass[self.id==ID], self.rvir[self.id==ID], self.rs[self.id==ID]
    
    def get_hosthalo(self, mass_min):
        host = (self.upid == -1) & (self.mass > mass_min)
        return self.pos[host], self.vel[host], self.mass[host], self.id[host], self.rvir[host], self.rs[host], self.sigmav[host]
        
def load_Uchuu(dir_path):
    ID = np.array([])
    x = np.array([])
    y = np.array([])
    z = np.array([])
        
    vx = np.array([])
    vy = np.array([])
    vz = np.array([])
        
    upid = np.array([])
    pid = np.array([])
        
    mass = np.array([])
        
    rvir = np.array([])
    rs = np.array([])
    
    sigmav = np.array([])
    
    for path in os.listdir(dir_path):
        filename = os.path.join(dir_path, path)
        with h5py.File(filename, 'r') as f:
            ID = np.append(ID, np.array(f['id'][:]))
            x = np.append(x, np.array(f['x'][:]))
            y = np.append(y, np.array(f['y'][:]))
            z = np.append(z, np.array(f['z'][:]))
        
            vx = np.append(vx, np.array(f['vx'][:]))
            vy = np.append(vy, np.array(f['vy'][:]))
            vz = np.append(vz, np.array(f['vz'][:]))
        
            upid = np.append(upid, np.array(f['upid'][:]))
            pid = np.append(pid, np.array(f['pid'][:]))
        
            mass = np.append(mass, np.array(f['Mvir'][:]))
        
            rvir = np.append(rvir, np.array(f['Rvir'][:]))
            rs = np.append(rs, np.array(f['rs'][:]))
            
            sigmav = np.append(sigmav, np.array(f['vrms'][:]))
        
    position = np.vstack([x,y,z]).T
    velocity = np.vstack([vx, vy, vz]).T
    
    return position, velocity, mass, ID, upid, pid, rvir, rs, sigmav
        
        
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
    
def apply_rsd_box(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    boxsize,
    redshift,
    cosmo,
    los="z",
):
    """
    Periodic simulation boxにplane-parallel RSDを適用する。

    Parameters
    ----------
    x, y, z : array-like
        Real-space Cartesian positions [Mpc/h].

    vx, vy, vz : array-like
        Physical peculiar velocities [km/s].

    boxsize : float
        Periodic box size [Mpc/h].

    redshift : float
        Snapshot redshift.

    cosmo : astropy.cosmology object
        Cosmology.

    los : {"x", "y", "z"}
        Fixed line-of-sight direction.
        論文に合わせる場合は "z"。

    Returns
    -------
    x_rsd, y_rsd, z_rsd : ndarray
        Redshift-space positions [Mpc/h].
    """
    x_rsd = np.asarray(x, dtype=np.float64).copy()
    y_rsd = np.asarray(y, dtype=np.float64).copy()
    z_rsd = np.asarray(z, dtype=np.float64).copy()

    velocities = {
        "x": np.asarray(vx, dtype=np.float64),
        "y": np.asarray(vy, dtype=np.float64),
        "z": np.asarray(vz, dtype=np.float64),
    }

    if los not in velocities:
        raise ValueError("los must be 'x', 'y', or 'z'.")

    a = 1.0 / (1.0 + redshift)
    h = cosmo.h

    # Astropyの H(z): km/s/Mpc
    H_mpc = cosmo.H(redshift).to_value(u.km / u.s / u.Mpc)

    # 座標が Mpc/h なので、変位も Mpc/h にする
    #
    # Δs [Mpc/h] = h * vpec / (a H)
    #              = vpec / (a * (H / h))
    delta_s = h * velocities[los] / (a * H_mpc)

    if los == "x":
        x_rsd += delta_s
    elif los == "y":
        y_rsd += delta_s
    else:
        z_rsd += delta_s

    # periodic boundaryに戻す
    x_rsd %= boxsize
    y_rsd %= boxsize
    z_rsd %= boxsize

    return x_rsd, y_rsd, z_rsd 
    