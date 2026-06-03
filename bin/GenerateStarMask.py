import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, vstack, unique, join
import astropy.io.ascii as ascii

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

from pfsimaging import imaging as Im
from pfsimaging import Loader as loader

from multiprocessing import Pool
from functools import partial

from scipy.spatial import cKDTree

autumn = None
gal_ra = None
gal_dec = None
gal_id = None

STAR_CACHE = {}

def _main():
    global autumn
    autumn = Im.TractPatch("spring")
    tract_data = autumn.data
    tract_list = autumn.get_tract()
    
    for tract in tract_list:
        gal_ra, gal_dec = GetRandoms(tract)

        if gal_ra is None:
            print(f"Random Points for tract {tract} does not exist")
            continue
            
        adj_tracts = get_adjacent_tracts(autumn.data, tract)
        args = [(adj, gal_ra, gal_dec) for adj in adj_tracts]
    
        with Pool(len(adj_tracts)) as p:
            results = p.map(process_tract, args)
        
        mask_halo = np.zeros(len(gal_ra), dtype=bool)
        mask_ghost = np.zeros(len(gal_ra), dtype=bool)
        mask_blooming = np.zeros(len(gal_ra), dtype=bool)
        
        for h, g, b in results:
            mask_halo |= h
            mask_ghost |= g
            mask_blooming |= b

        mask_file = f"/lustre/work/YukaYamada/data/PFS/s23b-bsmask/tracts/randoms/{tract}.fits"
        print(f"Saving bright star mask for tract {tract}") 
            
        cols = [
            fits.Column(name='ra', format='D', array=gal_ra), # float64
            fits.Column(name='dec', format='D', array=gal_dec), # float64
            fits.Column(name="halo", format="L", array=mask_halo),
            fits.Column(name="ghost", format="L", array=mask_ghost),
            fits.Column(name="blooming", format="L", array=mask_blooming),
        ]
        
        fits.BinTableHDU.from_columns(cols).writeto(mask_file, overwrite=True)

def main():
    #global autumn
    autumn = Im.TractPatch("autumn")
    tract_data = autumn.data
    tract_list = ascii.read("dat/november_tracts_autumn.csv")['tract']
    #tract_list = autumn.get_tract()
    #tract_list = [9579, 9580, 9336, 9337]
    
    for tract in tract_list:
        if os.path.exists(f"/lustre/work/YukaYamada/data/PFS/s23b-bsmask/tracts/galaxies/{tract}.fits"):
            print(f"already have bs mask for tract {tract}")
            continue
                          
        global gal_ra, gal_dec, gal_id
        gal_ra, gal_dec, gal_id = GetGalaxies(tract)

        if gal_ra is None:
            print(f"Galaxy file for tract {tract} does not exist")
            continue

        if len(gal_ra) == 0:
            print(f"No galaxies in tract {tract}")
            continue
            
        adj_tracts = get_adjacent_tracts(autumn.data, tract)
        args = [(adj, gal_ra, gal_dec) for adj in adj_tracts]
    
        with Pool(len(adj_tracts)) as p:
            results = p.map(process_tract, args)
        
        mask_halo = np.zeros(len(gal_ra), dtype=bool)
        mask_ghost = np.zeros(len(gal_ra), dtype=bool)
        mask_blooming = np.zeros(len(gal_ra), dtype=bool)
        
        for h, g, b in results:
            mask_halo |= h
            mask_ghost |= g
            mask_blooming |= b

        mask_file = f"/lustre/work/YukaYamada/data/PFS/s23b-bsmask/tracts/galaxies/{tract}.fits"
        print(f"Saving bright star mask for tract {tract}") 
            
        cols = [
            fits.Column(name='id', format='K', array=gal_id),   # int64
            fits.Column(name='ra', format='D', array=gal_ra), # float64
            fits.Column(name='dec', format='D', array=gal_dec), # float64
            fits.Column(name="halo", format="L", array=mask_halo),
            fits.Column(name="ghost", format="L", array=mask_ghost),
            fits.Column(name="blooming", format="L", array=mask_blooming),
        ]
        
        fits.BinTableHDU.from_columns(cols).writeto(mask_file, overwrite=True)
                
def process_tract(args):
    #書き込むのはこのtractだけ
    #星はadjacent tractもとってくる
    tract, gal_ra, gal_dec = args
    
    star_ra, star_dec, star_imag = GetStars(tract)
    
    if (star_ra is None):
        mask = np.zeros(len(gal_ra), dtype=bool)
        return mask, mask, mask

    _halo, _ghost, _blooming = GetMask_fast(star_ra, star_dec, star_imag, gal_ra, gal_dec)

    return _halo, _ghost, _blooming

def GetStars(tract):
    StarPath = f'/lustre/work/YukaYamada/data/Gaia_HSC/{tract}_stars.fits'
    if os.path.exists(StarPath):
        hdu = fits.open(StarPath)

        data = hdu[1].data
        ID = data['source_id']
        ra = data['ra']
        dec = data['dec']
        phot_g = data['phot_g_mean_mag']
        phot_bp = data['phot_bp_mean_mag']
        phot_rp = data['phot_rp_mean_mag']
        hdu.close()
        
        x = phot_bp - phot_rp
        imag = ( phot_g + 4.906063e-02 - 6.084751e-01 * x + 5.999354e-02 * x**2 + 8.071123e-03 * x**3 + 7.058538e-04 * x**4)
        
        return ra, dec, imag
    else:
        print(f"Stellar file for tract {tract} does not exist")
        return None, None, None
    
def GetGalaxies(tract):
    bs_pz_dir = "/lustre/work/jingjing.shi/pfs_co_fa/data_proc/s23b_wide/tracts/"
    #bs_pz_dir = "/lustre/work/YukaYamada/data/PFS/s23-colorterm/tracts_rdeep/"
    bs_pz_fn = os.path.join(bs_pz_dir, f"s23b_gal_{tract}.fits")
    if not os.path.exists(bs_pz_fn):
        print(f"Error: {bs_pz_fn} does not exist.")
        return None, None, None
    with fits.open(bs_pz_fn, memmap=False) as hdul:
        data = hdul[1].data
        ra = data["ra"]
        dec = data["dec"]
        galid = data["object_id"]
    return ra, dec, galid

def GetRandoms(tract):
    path = f"/lustre/work/YukaYamada/data/PFS/s23-colorterm/tracts_ran/{tract}.fits"
    if os.path.exists(path):
        with fits.open(path) as hdu:
            if len(hdu)==2:
                data = hdu[1].data
                ra = data['ra']%360
                dec = data['dec']
                return ra, dec
            else:
                return None, None
    else:
        return None, None

def get_adjacent_tracts(tract_dict, target_tract, tol=1.0):

    if target_tract not in tract_dict:
        raise ValueError(f"{target_tract} is not in tract_dict")

    target_center = np.array(tract_dict[target_tract]['center'])

    tract_ids = list(tract_dict.keys())
    centers = np.array([tract_dict[t]['center'] for t in tract_ids])

    # 差分
    dra_raw = np.abs(centers[:, 0] - target_center[0])
    dra = np.minimum(dra_raw, 360.0 - dra_raw)   # ⭐ ここが重要
    ddec = np.abs(centers[:, 1] - target_center[1])

    # ステップ幅推定
    nonzero_dra = np.sort(np.unique(np.round(dra[dra > 0], 6)))
    nonzero_ddec = np.sort(np.unique(np.round(ddec[ddec > 0], 6)))

    if len(nonzero_dra) == 0 or len(nonzero_ddec) == 0:
        return []

    step_ra = nonzero_dra[0]
    step_dec = nonzero_ddec[0]

    neighbors = []
    for t, dx, dy in zip(tract_ids, dra, ddec):

        cond_ra = (abs(dx - step_ra) < tol) or (dx < tol)
        cond_dec = (abs(dy - step_dec) < tol) or (dy < tol)

        if cond_ra and cond_dec:
            neighbors.append(t)

    return sorted(neighbors)

def GetMask(star_ra, star_dec, star_imag, gal_ra, gal_dec):
    mask_halo = np.zeros(len(gal_ra), dtype=bool)
    mask_ghost = np.zeros(len(gal_ra), dtype=bool)
    mask_blooming = np.zeros(len(gal_ra), dtype=bool)

    for _ra, _dec, _imag in zip(star_ra, star_dec, star_imag):
        halo = 1.105 * 1e3 * np.exp(-0.347 * _imag) + 4.950
        if _imag < 6.0:
            ghost = 700.0
        else:
            ghost = 108901*np.exp(-0.8624*_imag) + 74.105
    
        if(_imag < 8.75):
            blooming_size = 700.0
        else:
            blooming_size = -200.0 * _imag + 2460.0
        
        if(_imag < 6.75):
            blooming_width = 0.0
        else:
            blooming_width = -2.63*_imag + 34.93
        
        distance = np.sqrt((gal_ra - _ra)**2 + (gal_dec - _dec)**2)
        _halo = distance * 3600.0 < halo
        _ghost = distance * 3600.0 < ghost
        _blooming = (np.abs(gal_dec - _dec)*3600.0 < blooming_width)&(np.abs(gal_ra - _ra)*3600.0 < blooming_size)
    
        mask_halo |= _halo
        mask_ghost |= _ghost
        mask_blooming |= _blooming
    return mask_halo, mask_ghost, mask_blooming

def GetMask_fast(star_ra, star_dec, star_imag, gal_ra, gal_dec):
    star_ra = star_ra%360
    gal_ra = gal_ra%360
    #raを0~360に入れる
    if(((gal_ra.max() - star_ra.min())>180.0) or ((star_ra.max() - gal_ra.min())>180.0)):
        #ra=0.0を跨いでいる場合 -180~180に折り返す
        star_ra = np.where(star_ra>180.0, star_ra - 360.0, star_ra)
        gal_ra = np.where(gal_ra>180.0, gal_ra - 360.0, gal_ra)
        
    gal_pos = np.vstack([gal_ra, gal_dec]).T
    tree = cKDTree(gal_pos)

    mask_halo = np.zeros(len(gal_ra), dtype=bool)
    mask_ghost = np.zeros(len(gal_ra), dtype=bool)
    mask_blooming = np.zeros(len(gal_ra), dtype=bool)

    for ra, dec, imag in zip(star_ra, star_dec, star_imag):

        # ===== halo =====
        halo = 1.105e3 * np.exp(-0.347 * imag) + 4.950
        r_halo = halo / 3600.0
        idx = tree.query_ball_point([ra, dec], r=r_halo)
        idx = np.array(idx, dtype=int)
        mask_halo[idx] = True

        # ===== ghost =====
        if imag < 6.75:
            ghost = 700.0
        elif imag < 9.5:
            ghost = 13.49 * imag**2 - 338.32 * imag + 2061.89
        else:
            ghost = 0.0

        if ghost > 0:
            r_ghost = ghost / 3600.0
            idx = tree.query_ball_point([ra, dec], r=r_ghost)
            idx = np.array(idx, dtype=int)
            mask_ghost[idx] = True

        # ===== blooming =====
        if imag < 8.75:
            blooming_size = 700.0
        else:
            blooming_size = -200.0 * imag + 2460.0

        if imag < 6.75:
            blooming_width = 0.0
        else:
            blooming_width = -2.63 * imag + 34.93

        if blooming_width > 0:
            r = (blooming_size**2 + blooming_width**2)**0.5 / 3600.0
            idx = tree.query_ball_point([ra, dec], r=r)
            idx = np.array(idx, dtype=int)

            sub_ra = gal_ra[idx]
            sub_dec = gal_dec[idx]

            mask = (
                (np.abs(sub_dec - dec) * 3600.0 < blooming_width) &
                (np.abs(sub_ra - ra) * 3600.0 < blooming_size)
            )

            mask_blooming[idx[mask]] = True

    return mask_halo, mask_ghost, mask_blooming

if __name__ == '__main__':
	main()