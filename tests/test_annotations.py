import pytest

import dask
import dask.array as da
import numpy as np

from daskms import xds_from_ms
from daskms.experimental.zarr import xds_to_zarr

from predict.annotations import annotate_datasets, dim_propagator


def test_annotation_packing():
    with dask.annotate(dims=("row", "chan"), priority=2):
        A = da.ones(1000, chunks=100)

    with dask.annotate(priority=1):
        B = A + 1

    hlg = B.dask
    alayer = hlg.layers[A.name]
    blayer = hlg.layers[B.name]

    annots_a = alayer.__dask_distributed_annotations_pack__()
    annots_b = blayer.__dask_distributed_annotations_pack__()
    print(annots_a)
    annots = {}
    print(alayer.__dask_distributed_annotations_unpack__(annots, annots_a, alayer.keys()))
    print(annots)
    from dask.highlevelgraph import HighLevelGraph



@pytest.mark.parametrize("ms", ["/home/simon/data/WSRT_polar.MS_p0"])
def test_annotation_propagation(ms, tmp_path_factory):
    def vis(uvw, lm, freq):
        def _phase_delay(uvw, lm, freq):
            u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            l, m = lm[:, 0], lm[:, 1]
            n = np.sqrt(1.0 - l**2 - m**2)
            real_phase = u[:,None]*l[None,:] + v[:,None]*m[None,:] + w[:,None]*(n[None,:] - 1)
            real_phase = real_phase[:,:,None]*freq[None,None,:]/3e8
            return np.exp(-2 * np.pi * 1j * real_phase)

        pd = da.blockwise(_phase_delay, ("source", "row", "chan"),
                            uvw, ("row", "uvw"),
                            lm, ("source", "lm"),
                            freq, ("chan",),
                            dtype=uvw.dtype)
        return pd.sum(axis=0)

    out_datasets = []

    out_path = tmp_path_factory.mktemp("blah") / "out.zarr"
    datasets = xds_from_ms(ms, group_cols=["ANTENNA1"])

    with dask.config.set(array_plugins=[dim_propagator("row")]):
        datasets = annotate_datasets(datasets)

        for ds in datasets:
            dims = ds.dims
            chunks = ds.chunks

            with dask.annotate(dims=("source", "lm")):
                lm = da.ones((1000, 2), chunks=(100, 2))*1e-5

            with dask.annotate(dims=("chan",)):
                freq = da.linspace(.856e9, 2*.856e9,
                                   dims["chan"],
                                   chunks=chunks["chan"])

            data = vis(ds.UVW.data, lm, freq)

            out_ds = ds.assign(**{"MODEL_DATA": (("row", "chan"), data)})
            out_datasets.append(out_ds)

    writes = annotate_datasets(xds_to_zarr(out_datasets, out_path, ["MODEL_DATA"]))
    data = writes[-1].MODEL_DATA.data
    from pprint import pprint
    pprint(data.dask.layers[data.name].annotations)
