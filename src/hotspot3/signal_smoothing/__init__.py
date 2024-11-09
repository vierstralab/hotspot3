from hotspot3.signal_smoothing.modwt import *


def normalize_density(density, total_cutcounts):
    return (density / total_cutcounts * 1_000_000).astype(np.float32)