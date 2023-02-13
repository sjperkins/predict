import argparse
import warnings


from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import wsclean_predict

import dask
import dask.array as da
from daskms import xds_from_storage_ms, xds_from_storage_table, xds_to_storage_table
from daskms.experimental.zarr import xds_to_zarr
from daskms.fsspec_store import DaskMSStore

from predict.sky_model import WSCleanModel


def expand_vis(vis, corrs):
    if corrs == 1:
        return vis
    elif corrs == 2:
        return da.concatenate([vis, vis], axis=2).rechunk({2: corrs})
    elif corrs == 4:
        zeros = da.zeros_like(vis)
        return da.concatenate([vis, zeros, zeros, vis], axis=2).rechunk({2: corrs})
    else:
        raise ValueError(f"MS Correlations {corrs} not in (1, 2, 4)")


def predict_vis(args: argparse.Namespace, sky_model: WSCleanModel):
    datasets = xds_from_storage_ms(
        args.store,
        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={k: args.chunks[k] for k in ("row",)},
    )

    store = DaskMSStore(args.store)
    kw = {"group_cols": "__row__"} if store.type() == "casa" else {}

    ddid_ds = xds_from_storage_table(f"{args.store}::DATA_DESCRIPTION")
    pol_ds = xds_from_storage_table(f"{args.store}::POLARIZATION", **kw)
    spw_ds = xds_from_storage_table(f"{args.store}::SPECTRAL_WINDOW", **kw)
    field_ds = xds_from_storage_table(f"{args.store}::FIELD", **kw)

    (ddid_ds,) = dask.compute(ddid_ds)
    (pol_ds,) = dask.compute(pol_ds)
    (spw_ds,) = dask.compute(spw_ds)
    (field_ds,) = dask.compute(field_ds)

    out_datasets = []

    for ds in datasets:
        field = field_ds[ds.attrs["FIELD_ID"]]
        ddid = ddid_ds[ds.attrs["DATA_DESC_ID"]]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.values[0]]

        frequency = spw.CHAN_FREQ.data[0]
        # frequency = da.linspace(.856e9, 2*.856e9, 4096, chunks=args.chunks["chan"])

        with warnings.catch_warnings():
            # Ignore dask chunk warnings emitted when going from 1D
            # inputs to a 2D space of chunks
            warnings.simplefilter("ignore", category=da.PerformanceWarning)
            vis = wsclean_predict(
                ds.UVW.data,
                radec_to_lm(sky_model.radec, field.PHASE_DIR.values[0][0]),
                sky_model.source_type,
                sky_model.flux,
                sky_model.spi,
                sky_model.log_poly,
                sky_model.ref_freq,
                sky_model.gauss_shape,
                frequency,
            )

            vis = expand_vis(vis, pol.NUM_CORR.values[0])

        # Assign visibilities to MODEL_DATA array on the dataset
        ods = ds.assign(**{args.output_column: (("row", "chan", "corr"), vis)})
        out_datasets.append(ods)

    # Create a write to the table
    write = xds_to_zarr(out_datasets, args.output_store, columns=[args.output_column])
    # Add to the list of writes
    dask.compute(write, sync=True)
