from pycorr import TwoPointCorrelationFunction
from pycorr import KMeansSubsampler

import numpy as np

from Corrfunc.theory.wp import wp as wp_corr

def wp_lc(data_table, random_table, bins):
    """
    data: "ra", "dec", "dist", "weight"
    random: "ra", "dec", "dist", "weight"
    """
    
    RA = data_table["ra"]
    DEC = data_table["dec"]
    weight = data_table["weight"]
    CZ = data_table["dist"]
    
    if random_table is None:
        randoms_positions1 = None
        rand_weight = None
        estimator='auto'
    else:
        rand_RA = random_table["ra"]
        rand_DEC = random_table["dec"]
        rand_weight = random_table["weight"]
        rand_CZ = random_table["dist"]
        
        randoms_positions1 = np.vstack([rand_RA, rand_DEC, rand_CZ])
        estimator='landyszalay'
    
    
    data_positions1 = np.vstack([RA, DEC, CZ])

    edges = (bins, np.linspace(-40, 40, 81))
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=data_positions1, data_weights1=weight,
                                         randoms_positions1=randoms_positions1, randoms_weights1 = rand_weight,
                                         estimator=estimator, position_type="rdd",
                                         engine='corrfunc', nthreads=4)
    sep, _wp = result(pimax=None, return_sep=True)
    return sep, _wp

def wp_box(x, y, z, boxsize, bins, nthreads=1):
    pimax = 40.0

    results_wp = wp_corr(
        boxsize,
        nthreads,
        bins,
        pimax,
        x,
        y,
        z
    )

    return results_wp["wp"]

def split_jackknife(data_table, random_table, njacks = 128):
    """
    data: "ra", "dec", "dist", "weight"
    random: "ra", "dec", "dist", "weight"
    """
    RA = data_table["ra"]
    DEC = data_table["dec"]
    weight = data_table["weight"]
    CZ = data_table["dist"]

    lss_positions = [RA, DEC, CZ]

    rand_RA = random_table["ra"]
    rand_DEC = random_table["dec"]
    rand_weight = random_table["weight"]
    rand_CZ = random_table["dist"]
    rand_positions = [rand_RA, rand_DEC, rand_CZ]


    subsampler = KMeansSubsampler(
        mode="angular",
        positions=rand_positions,
        nsamples=njacks,
        nside=512,
        random_state=42,
        position_type="rdd",
    )

    lss_labels = subsampler.label(lss_positions)
    rand_labels = subsampler.label(rand_positions)
    return lss_labels, rand_labels
    

def wp_jackknife_lc(data_table, random_table, bins, njacks = 128):
    """
    data: "ra", "dec", "dist", "weight"
    random: "ra", "dec", "dist", "weight"
    """
    lss_labels, rand_labels = split_jackknife(data_table, random_table, njacks = njacks)
    
    RA = data_table["ra"]
    DEC = data_table["dec"]
    weight = data_table["weight"]
    CZ = data_table["dist"]
    data_positions1 = np.vstack([RA, DEC, CZ])

    rand_RA = random_table["ra"]
    rand_DEC = random_table["dec"]
    rand_weight = random_table["weight"]
    rand_CZ = random_table["dist"]
    randoms_positions1 = np.vstack([rand_RA, rand_DEC, rand_CZ])

    edges = (bins, np.linspace(-40, 40, 81))
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=data_positions1, data_weights1=weight,
                                         randoms_positions1=randoms_positions1, randoms_weights1 = rand_weight,
                                         data_samples1=lss_labels, randoms_samples1=rand_labels,
                                         estimator='landyszalay', position_type="rdd",
                                         engine='corrfunc', nthreads=4)

    rp, wp, cov = result.get_corr(pimax=None, return_sep=True, return_cov=True)
    std = np.diag(cov)**0.5
    return rp, wp, std
    
    
    