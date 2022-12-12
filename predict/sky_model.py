from argparse import Namespace
from collections import namedtuple

import dask.array as da

WSCleanModel = namedtuple(
    "WSCleanModel",
    ["source_type", "radec", "flux", "spi", "ref_freq", "log_poly", "gauss_shape"],
)


def generate_sky_model(args: Namespace) -> WSCleanModel:
    sdims = args.dimensions["source"]
    schunks = args.chunks["source"]

    # MeerKAT centre frequency
    ref_freq = da.full(sdims, (0.856e9 + 2 * 0.856e9) / 2, chunks=schunks)
    source_type = da.full(sdims, "POINT", chunks=schunks, dtype="<U5")
    flux = da.random.random(sdims, chunks=schunks) * 1e-4
    spi = da.random.random((sdims, 2), chunks=schunks) * 1e-3

    # six degrees around zero
    radec = da.random.random((sdims, 2), chunks=(schunks, 2))
    radec = da.deg2rad(6.0 * (radec - 0.5))
    log_si = da.full(sdims, False, chunks=schunks)
    gauss_shape = da.zeros((sdims, 3), chunks=schunks)

    return WSCleanModel(
        source_type,
        radec,
        flux,
        spi,
        ref_freq,
        log_si,
        gauss_shape,
    )
