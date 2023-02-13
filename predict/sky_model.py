from collections import namedtuple

import numpy as np

from africanus.model.wsclean.file_model import load

WSCleanModel = namedtuple(
    "WSCleanModel",
    ["source_type", "radec", "flux", "spi", "ref_freq", "log_poly", "gauss_shape"],
)


def load_sky_model(sky_model_filename: str) -> WSCleanModel:
    wsclean_comps = {
        column: np.asarray(values) for column, values in load(sky_model_filename)
    }

    if np.unique(wsclean_comps["LogarithmicSI"]).shape[0] > 1:
        raise ValueError(
            f"Mixed log and ordinary polynomial "
            f"coefficients in '{sky_model_filename}'"
        )

    # Create radec array
    radec = np.concatenate(
        (wsclean_comps["Ra"][:, None], wsclean_comps["Dec"][:, None]), axis=1
    )

    # Create gaussian shapes
    gauss_shape = np.stack(
        (
            wsclean_comps["MajorAxis"],
            wsclean_comps["MinorAxis"],
            wsclean_comps["Orientation"],
        ),
        axis=-1,
    )

    return WSCleanModel(
        wsclean_comps["Type"],
        radec,
        wsclean_comps["I"],
        wsclean_comps["SpectralIndex"],
        wsclean_comps["ReferenceFrequency"],
        wsclean_comps["LogarithmicSI"],
        gauss_shape,
    )
