from pycorr import TwoPointCorrelationFunction
import numpy as np

def wp(data_table, random_table, bins):
    """
    data: "ra", "dec", "dist", "weight"
    random: "ra", "dec", "dist", "weight"
    """
    
    RA = data_table["ra"]
    DEC = data_table["dec"]
    weight = data_table["weight"]
    CZ = data_table["dist"]
    
    rand_RA = random_table["ra"]
    rand_DEC = random_table["dec"]
    rand_weight = random_table["weight"]
    rand_CZ = random_table["dist"]
    
    
    data_positions1 = np.vstack([RA, DEC, CZ])
    randoms_positions1 = np.vstack([rand_RA, rand_DEC, rand_CZ])

    edges = (bins, np.linspace(-40, 40, 81))
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=data_positions1, data_weights1=weight,
                                         randoms_positions1=randoms_positions1, randoms_weights1 = rand_weight,
                                         estimator="landyszalay", position_type="rdd",
                                         engine='corrfunc', nthreads=4)
    sep, wp = result(pimax=None, return_sep=True)
    return sep, wp
    
    