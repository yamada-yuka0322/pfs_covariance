from astropy.table import Table

from pfscovariance import loader, hod, clustering

from astropy.cosmology import Planck18

from scipy.optimize import differential_evolution

import numpy as np

import matplotlib.pyplot as plt

import time

def calculate_clustering(lss, randoms):
    rand_ra = np.asarray(randoms["RA"], dtype=np.float64)
    rand_dec = np.asarray(randoms["DEC"], dtype=np.float64)
    rand_z = np.asarray(randoms["Z"], dtype=np.float64)

    rand_dist = (
        Planck18.comoving_distance(rand_z).value
        * Planck18.h
    )

    rand_weight = np.asarray(
        randoms["WEIGHT"],
        dtype=np.float64,
    )

    random_table = Table({
        "ra": rand_ra,
        "dec": rand_dec,
        "dist": rand_dist,
        "weight": rand_weight,
    })

    ra = np.asarray(lss["RA"], dtype=np.float64)
    dec = np.asarray(lss["DEC"], dtype=np.float64)
    z = np.asarray(lss["Z"], dtype=np.float64)

    dist = (
        Planck18.comoving_distance(z).value
        * Planck18.h
    )

    weight = np.asarray(
        lss["WEIGHT"],
        dtype=np.float64,
    )

    data_table = Table({
        "ra": ra,
        "dec": dec,
        "dist": dist,
        "weight": weight,
    })

    rbins = np.logspace(
        np.log10(0.04),
        np.log10(32.0),
        18,
    )

    rp, wp, err = clustering.wp_jackknife_lc(
        data_table,
        random_table,
        rbins,
    )

    return (
        np.asarray(rp, dtype=np.float64),
        np.asarray(wp, dtype=np.float64),
        np.asarray(err, dtype=np.float64),
    )

def calculate_hod_wp(
    params,
    halo_table,
    particle_mass,
    redshift,
    rbins,
    boxsize=2000.0,
):
    """
    HOD parameterからmockを生成し、redshift-space wpを計算する。

    params
    ------
    logMc, sigma_m, As, logM0, logM1_, alpha, fv
    """
    t0 = time.perf_counter()
    
    (
        logMc,
        sigma_m,
        As,
        logM0,
        #logM1_,
        alpha,
        fv,
    ) = params

    params_cent = {
        "logMc": logMc,
        "sigma_M": sigma_m,
        "Ac": 1.0,
    }

    params_sat = {
        "As": As,
        "logM0": logM0,
        #"logM1_": logM1_,
        "alpha": alpha,
        "fv": fv,
    }

    galaxies = hod.populate_galaxies(
        halo_table,
        redshift,
        boxsize,
        particle_mass,
        params_cent,
        params_sat,
        np_min=65,
    )
    
    t1 = time.perf_counter()

    # HODによって銀河が生成されなかった場合
    if galaxies is None or len(galaxies) < 2:
        raise ValueError("Too few galaxies were populated.")

    x = np.asarray(galaxies["x"], dtype=np.float64)
    y = np.asarray(galaxies["y"], dtype=np.float64)
    z = np.asarray(galaxies["z"], dtype=np.float64)

    vx = np.asarray(galaxies["vx"], dtype=np.float64)
    vy = np.asarray(galaxies["vy"], dtype=np.float64)
    vz = np.asarray(galaxies["vz"], dtype=np.float64)

    x_rsd, y_rsd, z_rsd = loader.apply_rsd_box(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        boxsize=boxsize,
        redshift=redshift,
        cosmo=Planck18,
        los="z",
    )
    
    t2 = time.perf_counter()

    wp_model = clustering.wp_box(
        x_rsd,
        y_rsd,
        z_rsd,
        boxsize,
        rbins,
    )
    
    t3 = time.perf_counter()

    wp_model = np.asarray(wp_model, dtype=np.float64)

    number_density = len(galaxies) / boxsize**3
    
    print(
        f"Ngal={len(galaxies):,}, "
        f"populate={t1-t0:.2f}s, "
        f"RSD={t2-t1:.2f}s, "
        f"wp={t3-t2:.2f}s, "
        f"total={t3-t0:.2f}s"
    )

    return wp_model, number_density
    
def objective_function(
    params,
    rp_data,
    wp_data,
    wp_err,
    halo_table,
    particle_mass,
    redshift,
    rbins,
    boxsize=2000.0,
    target_density=None,
    density_error=None,
    verbose=False,
):
    try:
        wp_model, model_density = calculate_hod_wp(
            params=params,
            halo_table=halo_table,
            particle_mass=particle_mass,
            redshift=redshift,
            rbins=rbins,
            boxsize=boxsize
        )
    except Exception as exc:
        if verbose:
            print("Failed parameters:", params)
            print(exc)

        return 1.0e30

    valid = (
        np.isfinite(wp_data)
        & np.isfinite(wp_model)
        & np.isfinite(wp_err)
        & (wp_err > 0.0)
    )

    if np.count_nonzero(valid) == 0:
        return 1.0e30

    normalized_residual = (
        wp_model[valid] - wp_data[valid]
    ) / wp_err[valid]

    chi2_wp = np.sum(normalized_residual**2)

    chi2_density = 0.0

    if (
        target_density is not None
        and density_error is not None
        and density_error > 0.0
    ):
        chi2_density = (
            (model_density - target_density)
            / density_error
        )**2

    chi2_total = chi2_wp + chi2_density

    ndata = np.count_nonzero(valid)

    if target_density is not None:
        ndata += 1

    rms = np.sqrt(chi2_total / ndata)

    if verbose:
        print(
            f"rms={rms:.4f}, "
            f"chi2_wp={chi2_wp:.2f}, "
            f"chi2_density={chi2_density:.2f}, "
            f"nbar={model_density:.4e}"
        )

    return rms

def fit_params(
    rp_data,
    wp_data,
    wp_err,
    halo_table,
    particle_mass,
    redshift,
    boxsize=2000.0,
    target_density=None,
    density_error=None,
):
    rbins = np.logspace(
        np.log10(0.04),
        np.log10(32.0),
        18,
    )

    bounds = [
        (11.0, 12.5),   # logMc
        (0.01, 0.3),    # sigma_m
        (0.01, 0.15),   # As
        (11.0, 12.5),   # logM0
        #(2.0, 10.0),   # logM1_
        (-1.0, 1.0),     # alpha
        (0.2, 2.0),     # fv
    ]

    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(
            rp_data,
            wp_data,
            wp_err,
            halo_table,
            particle_mass,
            redshift,
            rbins,
            boxsize,
            target_density,
            density_error,
            True,
        ),
        strategy="best1bin",
        popsize=4,
        maxiter=15,
        tol=1.0e-2,
        polish=False,
        workers=1,
        updating="immediate",
        seed=42,
        disp=True,
    )

    return result
    
def main():
    lss = loader.load_lss(
        "/lustre/work/YukaYamada/data/DESI/clustering/"
        "ELG_N_clustering.dat.fits"
    )

    randoms = loader.load_random(
        "/lustre/work/YukaYamada/data/DESI/clustering/random/"
        "ELG_N_0_clustering.ran.fits"
    )

    rp_data, wp_data, wp_err = calculate_clustering(
        lss,
        randoms,
    )
    path="/lustre/work/ryuichiro.hada/PFS/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z1.100/halo_info"
    redshift = 1.1
    
    Halo = loader.Halo(path, Type='Abacus')
    
    if (Halo.halocat is not None):
        print("Finished loading halo catalog")

    target_density = None
    density_error = None
    boxsize=500.0

    result = fit_params(
        rp_data=rp_data,
        wp_data=wp_data,
        wp_err=wp_err,
        halo_table=Halo.halocat,
        particle_mass=Halo.pm,
        redshift=redshift,
        boxsize=boxsize,
        target_density=target_density,
        density_error=density_error,
    )

    names = [
        "logMc",
        "sigma_m",
        "As",
        "logM0",
        #"logM1_",
        "alpha",
        "fv",
    ]

    print("\nBest-fit result")
    print("----------------")

    for name, value in zip(names, result.x):
        print(f"{name:8s} = {value:.6f}")

    print(f"minimum objective = {result.fun:.6f}")
    print(f"success = {result.success}")
    print(f"message = {result.message}")

    rbins = np.logspace(
        np.log10(0.04),
        np.log10(32.0),
        18,
    )

    wp_best, density_best = calculate_hod_wp(
        params=result.x,
        halo_table=Halo.halocat,
        particle_mass=Halo.pm,
        redshift=redshift,
        rbins=rbins,
        boxsize=boxsize
    )

    output = Table({
        "rp": rp_data,
        "wp_data": wp_data,
        "wp_error": wp_err,
        "wp_bestfit": wp_best,
    })
    
    ax = plt.gca()
    ax.errorbar(rp_data, rp_data*wp_data, yerr=wp_err * rp_data, fmt='o')
    ax.plot(rp_data, wp_best*rp_data)
    ax.set_xlabel(r'$r_{p}$')
    ax.set_ylabel(r'$r_{p} w_{p}(r_{p})$')
    ax.set_xscale("log")
    ax.grid(True)
    ax.set_xlim(0.04, 32)
    ax.set_ylim(10, 80)
    plt.savefig('../figure/HOD.pdf')
    plt.show()

    #output.write(
        #"hod_bestfit_wp.fits",
        #overwrite=True,
    #)

    #print(f"Best-fit number density = {density_best:.6e}")
    
    
if __name__ == '__main__':
    main()