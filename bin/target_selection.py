'''
Target selection for the SSP_CO cosmology targets.
'''
from utils.common_imports import *
from astropy.table import Table, vstack, join
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import pandas as pd
import time
import os

def get_extinction_coeff():
    """Get the extinction coefficients for HSC filters."""
    # absorption coefficients of HSC filters
    absorptionCoeff = {
        "g"    : 3.240,
        "r"    : 2.276,
        "i"    : 1.633,
        "z"    : 1.263,
        "y"    : 1.075,
        "n387" : 4.007,
        "n468" : 3.351,
        "n515" : 2.939,
        "n527" : 2.855,
        "n656" : 2.077,
        "n718" : 1.812,
        "n816" : 1.458,
        "n921" : 1.187,
        "n973" : 1.083,
        "n1010": 1.013,
        "i945" : 1.134,
    }
    return absorptionCoeff
    
def get_dust_map_csfd_desi_equatorial():
    """Get the CSFD DESI equatorial dust map data."""

    # Yi-Kuan's equatorial dust map
    dustmap_file = "../data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial.fits"
    hdul = fits.open(dustmap_file)
    df_csfd_desi_equatorial = hdul[1].data
    df_csfd_desi_eq = df_csfd_desi_equatorial['EBV_CSFD_DESI_merged_at_1deg']   

    return df_csfd_desi_eq


def add_dust_attenuation(galaxy, absorptionCoeff, dustmap='csfd_desi', nside=512):
    '''
    Calculate the dust attenuation at given RA and Dec using specified dust map.
    Parameters:
    galaxy : astropy Table
        Table containing 'ra' and 'dec' columns.
    absorptionCoeff : dict
        Dictionary of absorption coefficients for different bands.
    dustmap : str, optional
        The dust map to use ('sfd', 'csfd_desi'). Default is 'csfd_desi'.
    nside : int, optional
        The nside parameter for HEALPix. Default is 512.
    Returns:
    galaxy : astropy Table
        Table with added columns for dust attenuation in g, r, i, z, y bands.
    '''
    
    ra = galaxy['ra']
    dec = galaxy['dec']
    coord = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")

    if dustmap == 'sfd':
        sfd = SFDQuery()
        ebv_sfd = sfd(coord)
        a_g = absorptionCoeff['g'] * ebv_sfd
        a_r = absorptionCoeff['r'] * ebv_sfd
        a_i = absorptionCoeff['i'] * ebv_sfd
        a_z = absorptionCoeff['z'] * ebv_sfd
        a_y = absorptionCoeff['y'] * ebv_sfd
    elif dustmap == 'csfd_desi':
        df_csfd_desi_eq = get_dust_map_csfd_desi_equatorial()
        ebv_csfd_desi = df_csfd_desi_eq[hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)]
        a_g = absorptionCoeff['g'] * ebv_csfd_desi
        a_r = absorptionCoeff['r'] * ebv_csfd_desi
        a_i = absorptionCoeff['i'] * ebv_csfd_desi
        a_z = absorptionCoeff['z'] * ebv_csfd_desi
        a_y = absorptionCoeff['y'] * ebv_csfd_desi

    galaxy['a_g'] = a_g
    galaxy['a_r'] = a_r
    galaxy['a_i'] = a_i
    galaxy['a_z'] = a_z
    galaxy['a_y'] = a_y

    return galaxy


def _flux_to_mag(flux_nJy):
    '''
    Convert flux in nJy to magnitude.
    Parameters:
    flux_nJy : array-like
        Flux in nJy.
    Returns:
    mag : array-like
        Magnitude.
    '''
    
    flux_nJy = np.array(flux_nJy, dtype=np.float64)  
    mag = np.full_like(flux_nJy, np.nan) 
    
    mask = flux_nJy > 0  
    mag[mask] = -2.5 * np.log10(flux_nJy[mask] * 1e-32) - 48.6

    return mag

def get_tracts_s23b_wide():
    '''
    Get the tracts for s23b_wide spring and autumn fields.
    Returns:
    tract_spring : astropy Table
        Table of tracts in the spring field.
    tract_autumn : astropy Table
        Table of tracts in the autumn field.
    '''

    # define ra, dec ranges for spring, autumn, hectomap fields
    # ref. https://hsc.mtk.nao.ac.jp/ssp/survey/
    ra_spring_min, ra_spring_max = 120, 230
    ra_autumn_min, ra_autumn_max = -40, 50
    ra_hectomap_min, ra_hectomap_max = 198, 255
    dec_spring_min, dec_spring_max = -10, 10
    dec_autumn_min, dec_autumn_max = -10, 10
    dec_hectomap_min, dec_hectomap_max = 41, 46

    # read s23b_wide patch_qa.fits
    patch_qa = Table.read('../data_raw/s23b_wide/patch_qa.fits')

    # spring
    mask_spring = (patch_qa['ra'] > ra_spring_min) & (patch_qa['ra'] < ra_spring_max) & (patch_qa['dec'] > dec_spring_min) & (patch_qa['dec'] < dec_spring_max)
    tract_spring = np.unique(patch_qa['tract'][mask_spring])
    df = pd.DataFrame(tract_spring, columns=['tract'])
    df.to_csv("../data_proc/s23b_wide/tract_spring.csv", index=False)
    #tract_spring = Table.read("../data_proc/s23b_wide/tract_spring.csv")

    # autumn
    patch_qa_ra = patch_qa['ra']
    mask_tmp = patch_qa_ra > 330.
    patch_qa_ra[mask_tmp] -= 360.
    patch_qa_dec = patch_qa['dec']
    mask_autumn = (patch_qa_ra > ra_autumn_min) & (patch_qa_ra < ra_autumn_max) & (patch_qa_dec > dec_autumn_min) & (patch_qa_dec < dec_autumn_max) & (patch_qa['tract'] > 0)
    tract_autumn = np.unique(patch_qa['tract'][mask_autumn])
    df = pd.DataFrame(tract_autumn, columns=['tract'])
    df.to_csv("../data_proc/s23b_wide/tract_autumn.csv", index=False)
    #tract_autumn = Table.read("../data_proc/s23b_wide/tract_autumn.csv")

    # hectomap
    mask_hectomap = (patch_qa['ra'] > ra_hectomap_min) & (patch_qa['ra'] < ra_hectomap_max) & (patch_qa['dec'] > dec_hectomap_min) & (patch_qa['dec'] < dec_hectomap_max)
    tract_hectomap = np.unique(patch_qa['tract'][mask_hectomap])
    df = pd.DataFrame(tract_hectomap, columns=['tract'])
    df.to_csv("../data_proc/s23b_wide/tract_hectomap.csv", index=False)
    #tract_hectomap = Table.read("../data_proc/s23b_wide/tract_hectomap.csv")

    return tract_spring, tract_autumn, tract_hectomap


def read_tracts_s23b_wide():
    '''
    Read the tracts for s23b_wide spring and autumn fields from CSV files.
    Returns:
    tract_spring : astropy Table
        Table of tracts in the spring field.
    tract_autumn : astropy Table
        Table of tracts in the autumn field.
    '''
    tract_spring = Table.read("../data_proc/s23b_wide/tract_spring.csv")
    tract_autumn = Table.read("../data_proc/s23b_wide/tract_autumn.csv")
    tract_hectomap = Table.read("../data_proc/s23b_wide/tract_hectomap.csv")

    return tract_spring['tract'], tract_autumn['tract'], tract_hectomap['tract']


def get_bright_star_mask_pz(tract_list, field_name):
    '''
    Get the bright star mask and photoz of mizuki for each field.
    Parameters:
    tract_list : list
        List of tracts in the field.
    field_name : str
        Name of the field ('spring', 'autumn', 'hectomap').
    Returns:
    bs_pz_field : astropy Table
        Table of objects with bright star mask and photo-z info in the field.
    '''

    bs_pz_dir = "../data_raw/s23b-bsmask/tracts/"
    tables = [] 

    for tract in tract_list['tract']:
        bs_pz_fn = os.path.join(bs_pz_dir, f"{tract}.fits")

        if not os.path.exists(bs_pz_fn):
            print(f"{bs_pz_fn} does not exist.")
            continue
        with fits.open(bs_pz_fn, memmap=False) as hdul:
            data = hdul[1].data

            # Build mask more compactly (avoid repeated lookups)
            mask_brightstar = ~(
                data['g_mask_brightstar_halo'] |
                data['r_mask_brightstar_halo'] |
                data['i_mask_brightstar_halo'] |
                data['z_mask_brightstar_halo'] |
                data['g_mask_brightstar_ghost'] |
                data['r_mask_brightstar_ghost'] |
                data['i_mask_brightstar_ghost'] |
                data['z_mask_brightstar_ghost']
            )

            tables.append(Table(data[mask_brightstar])[[
                'object_id',
                'pz_best_mizuki',
                'pz_best_mizuki_isnull',
                'pz_risk_best_mizuki'
            ]])

    bs_pz = vstack(tables, metadata_conflicts='silent')
    print(f"There are {len(bs_pz)} objects with bright star mask and photo-z info in {field_name} field.")

    bs_pz.write(f'../data_proc/bs_pz_{field_name}.fits', overwrite=True)
    print(f'bs_pz_{field_name}.fits saved.')

    return bs_pz


def apply_bright_star_mask_pz(tract, galaxy):
    '''
    Apply the bright star mask and add photoz of mizuki for the tract.
    Parameters:
    tract : int
        The tract number.
    galaxy : astropy Table
        The galaxy catalog for the tract.
    '''
    
    bs_pz_dir = "../data_raw/s23b-bsmask/tracts/"
    bs_pz_fn = os.path.join(bs_pz_dir, f"{tract}.fits")
    if not os.path.exists(bs_pz_fn):
        print(f"Error: {bs_pz_fn} does not exist.")
        exit(1)
    with fits.open(bs_pz_fn, memmap=False) as hdul:
        data = hdul[1].data

    # Build mask more compactly (avoid repeated lookups)
    mask_brightstar = ~(
        data['g_mask_brightstar_halo'] |
        data['r_mask_brightstar_halo'] |
        data['i_mask_brightstar_halo'] |
        data['z_mask_brightstar_halo'] |
        data['g_mask_brightstar_ghost'] |
        data['r_mask_brightstar_ghost'] |
        data['i_mask_brightstar_ghost'] |
        data['z_mask_brightstar_ghost']
    )

    bs_pz_tract = Table(data[mask_brightstar])[[
        'object_id',
        'pz_best_mizuki',
        'pz_best_mizuki_isnull',
        'pz_risk_best_mizuki'
    ]]

    galaxy_joined = join(galaxy, bs_pz_tract, keys='object_id', join_type='left')

    return galaxy_joined


def target_selection(galaxy, zeropoint=False):
    '''
    Apply the target selection cuts to the hsc galaxy catalog.

    Parameters:
    galaxy : astropy Table
        The input galaxy catalog.
    zeropoint : bool, optional
        Whether to apply zeropoint correction using stellar offset. Default is False.

    Returns:
    mask : array-like
        Boolean mask indicating selected targets.

    NOTE: dust attenuation only applied to mag + color cuts
    TODO: bright galaxy mask
    '''

    # apply dust correction to forced cmodel magnitudes only
    g_mag = galaxy['g_cmodel_mag'] - galaxy['a_g']
    r_mag = galaxy['r_cmodel_mag'] - galaxy['a_r']
    i_mag = galaxy['i_cmodel_mag'] - galaxy['a_i']
    z_mag = galaxy['z_cmodel_mag'] - galaxy['a_z']


    # psf_flux_flag -> psf_flux is required by the observatory
    mask_psf_flag = (~galaxy['g_psf_flag']) & (~galaxy['r_psf_flag']) & (~galaxy['i_psf_flag']) &\
                    (~galaxy['z_psf_flag'])

    # mask_psf_flag_y = (~galaxy['y_psf_flag'])

    # deblend_skipped
    mask_deblend_skipped = (~galaxy['deblend_skipped'])
    
    # snr cuts
    mask_sn = (galaxy['g_cmodel_mag_err']<0.05*galaxy['g_cmodel_mag']-1.1)

    # extendedness cut of meas. 
    mask_ext = (galaxy['i_cmodel_mag_meas']-galaxy['i_psf_mag_meas']<-0.08) &\
               (~galaxy['i_cmodel_flag_meas']) & (~galaxy['i_psf_flag_meas'])
    
    # magnitude cuts
    mask_mag = (i_mag>22.5) & (i_mag<23.5) & (g_mag<24.0)

    # low surface brightness object cut (ref. Li Xiangchong et al. 2022, Table 2)
    mask_low_sb = (galaxy['i_apertureflux_10_mag']<=25.5) & (~galaxy['i_apertureflux_10_flag'])
    
    # extreme color cuts
    mask_extreme_color = (g_mag - r_mag < 1.5) & (i_mag - z_mag < 1.5) & \
                         (g_mag - r_mag > -1.5) & (i_mag - z_mag > -1.5)

    # color cuts
    cut1 = (g_mag - r_mag < 0.35)
    cut2 = (i_mag - z_mag > 2.0 * (g_mag - r_mag) - 0.55)
    cut3 = (i_mag - z_mag > 0.0)
    mask_color = (cut1 | cut2) & cut3

    # check if flux is smaller than 0 after mask
    # mask_neg_flux = (galaxy['g_psf_flux'][mask]<=0) | (galaxy['r_psf_flux'][mask]<=0) |\
    #                (galaxy['i_psf_flux'][mask]<=0) | (galaxy['z_psf_flux'][mask]<=0)


    # combine all masks
    mask = mask_psf_flag & mask_sn & mask_ext & mask_mag & mask_low_sb & mask_color & mask_extreme_color & mask_deblend_skipped    
    
    return mask



def main(field_name, tract_list, fn_out):
    '''
    Get the HSC galaxy catalog for a specific field and list of tracts.
    Parameters:
    field_name : str
        Name of the field ('spring', 'autumn', 'hectomap').
    tract_list : array-like
        List of tracts to include.
    fn_out : str
        Output filename for the combined ssp_co targets catalog.
    '''
    
    targets_catalog = Table()

    for tract in tract_list:
        file_path =  f"../data_proc/s23b_wide/tracts/s23b_gal_{tract}.fits"
        if os.path.exists(file_path):
            hdul = fits.open(file_path, memmap=True)
        else:
            print(f"Note: File '{file_path}' does not exist.")
            continue

        galaxy = Table(hdul[1].data)

        if len(galaxy)<1:
            continue

        # get dust attenuation for CSFD_DESI dust map
        absorptionCoeff = get_extinction_coeff()
        galaxy = add_dust_attenuation(galaxy, absorptionCoeff, dustmap='csfd_desi', nside=2048)

        # apply bright star mask and add photo-z info
        galaxy = apply_bright_star_mask_pz(tract, galaxy)

        # compute i_apertureflux_10_mag
        galaxy['i_apertureflux_10_mag'] = _flux_to_mag(galaxy['i_apertureflux_10_flux'])

        # apply target selection cuts 
        mask_ts = target_selection(galaxy, zeropoint=False)
        targets_catalog = vstack([targets_catalog, galaxy[mask_ts]])

        print(f'tract {tract} in field {field_name} done: {np.sum(mask_ts)}/({len(galaxy)}) targets selected.')

    targets_catalog.write(f'{fn_out}', overwrite=True)


if __name__ == '__main__':
    # define fields and tracts
    #tract_spring, tract_autumn, tract_hectomap = read_tracts_s23b_wide()
    tract_spring, tract_autumn, tract_hectomap = get_tracts_s23b_wide()
    t0 = time.time()
    main('spring', tract_spring, '../data_proc/s23b_wide/ssp_co_targets/s23b_spring.fits')
    t1 = time.time()
    print(f"Time taken for spring: {t1 - t0} seconds")
    main('autumn', tract_autumn, '../data_proc/s23b_wide/ssp_co_targets/s23b_autumn.fits')
    t2 = time.time()
    print(f"Time taken for autumn: {t2 - t1} seconds")
    #main('hectomap', tract_hectomap, '../data_proc/s23b_wide/ssp_co_targets/s23b_hectomap.fits')
    #t3 = time.time()
    #print(f"Time taken for hectomap: {t3 - t2} seconds")
    
    # combine all fields
    spring = Table.read('../data_proc/s23b_wide/ssp_co_targets/s23b_spring.fits')
    autumn = Table.read('../data_proc/s23b_wide/ssp_co_targets/s23b_autumn.fits')
    #hectomap = Table.read('../data_proc/s23b_wide/ssp_co_targets/s23b_hectomap.fits')  
    ssp_co_targets = vstack([spring, autumn])
    ssp_co_targets.write('../data_proc/s23b_wide/ssp_co_targets/s23b_ssp_co_targets.fits', overwrite=True)