from functools import partial, reduce
from tlz.functoolz import Compose

import dask
from dask.highlevelgraph import HighLevelGraph
from dask.blockwise import Blockwise

from typing import Iterable, Union
from uuid import uuid4

def annotate_datasets(datasets):
    """Annotate dask arrays on datasets with:

    1. Their xarray dimension schema. e.g. ("row", "chan", "corr").
    2. The dataset id.
    3. The number of blocks in each dimension.

    Parameters
    ----------

    datasets: list of xarray datasets

    Returns
    -------
    datasets: list of xarray datasets
        The original list of datasets with annotations
        added to the wrapped dask arrays
    """
    ds_blocks = [{d: len(c) for d, c in ds.chunks.items()} for ds in datasets]
    row_blocks = [blocks["row"] for blocks in ds_blocks]
    total_row_blocks = sum(row_blocks)
    start_row_block = reduce(lambda a, v: a + [a[-1] + v], row_blocks, [0])

    for i, (ds, blocks) in enumerate(zip(datasets, ds_blocks)):
        block_extents = {d: (0, b) for d, b in blocks.items()}
        block_extents["row"] = (start_row_block[i], total_row_blocks)

        for v in ds.data_vars.values():
            if dask.is_dask_collection(v.data):
                layer =  v.data.dask.layers[v.data.name]
                annotations = {
                    "dims": v.dims,
                    "dataset_id": i,
                    "blocks": block_extents
                }

                if layer.annotations:
                    layer.annotations.update(annotations)
                else:
                    layer.annotations = annotations

    return datasets


def is_key_arraylike(key):
    """Return True if key is a tuple of the form :code:`(str, int, ...)"""
    return (isinstance(key, tuple) and
            isinstance(key[0], str) and
            all(isinstance(item, int) for item in key[1:]))

def reduction_axes(layer):
    """Return the axis/keepdims parameters of any reduction methods, or False """
    try:
        k, task = next(iter(layer.items()))
    except StopIteration:
        return False

    # Sanity-checks
    if not is_key_arraylike(k) or not isinstance(task, tuple):
        return False

    fns = task[0].funcs if isinstance(task[0], Compose) else task[:1]

    try:
        pfn = next(iter(fn for fn in fns if isinstance(fn, partial)))
    except StopIteration as e:
        return False

    try:
        return tuple(pfn.keywords[kw] for kw in ("axis", "keepdims"))
    except KeyError as e:
        return False


def dummy_dimensions(ndims):
    return tuple(f"dummy-{uuid4().hex[:8]}" for _ in range(ndims))


def materialized_layer_dims(layer):
    """ Get or generate dimensions of the materialized layer """
    try:
        # May already be present
        return layer.annotations["dims"]
    except (KeyError, TypeError):
        # Generate dummy dimensions from the key length
        try:
            k = next(iter(layer.keys()))
        except StopIteration:
            return False

        # Needs to be array like
        if not is_key_arraylike(k):
            return False

        return dummy_dimensions(len(k[1:]))

def dims_from_ancestors(layer, ancestor_dims):
    """Returns a :code:`{i: d}` dict mapping layer indices to dimension names """
    index_dims = {}

    for ancestor_name, indices in layer.indices:
        if indices is None:  # Scalar?
            continue

        # Get ancestor dimensions and assign them
        # to the appropriate indices
        try:
            dims = ancestor_dims[ancestor_name]
        except KeyError:
            # References a dask.blockwise.BlockwiseDep,
            # rather than an actual dask collection
            # (i.e. a dask.array.Array)
            if ancestor_name in layer.io_deps:
                dims = dummy_dimensions(len(indices))
            else:
                raise KeyError(f"{ancestor_name} is not a valid dependency of {layer}")
        if len(dims) != len(indices):
            raise ValueError(f"Dimension lengths {dims} != {indices}")

        index_dims.update(zip(indices, dims))

    return index_dims

def dim_propagator(dims: Union[str, Iterable[str]]):
    def propagator(x):
        hlg = x.dask

        if not isinstance(hlg, HighLevelGraph):
            raise TypeError(f"{x} is not backed by a HighLevelGraph {type(hlg)}")

        layer = hlg.layers[x.name]
        dep_layer_dims = {}
        output_dims = None
        dataset_id = None
        ds_blocks = None

        # Obtain the dimension schema of ancestor layers
        # Generate them if they don't exist
        for dep_layer_name in hlg.dependencies[x.name]:
            dep_layer = hlg.layers[dep_layer_name]

            if isinstance(dep_layer, Blockwise):
                ndims = len(dep_layer.output_indices)
            elif dep_layer.is_materialized():
                k = next(iter(dep_layer.get_output_keys()))
                assert is_key_arraylike(k)
                ndims = len(k[1:])
            else:
                raise NotImplementedError(f"Layer type {dep_layer.layer_info_dict()['layer_type']}")

            # Get any existing layer annotations on the ancestor layer
            layer_annots = dep_layer.annotations or {}

            # Get any dims on the ancestor, or generate some random dims
            try:
                layer_dims = layer_annots["dims"]
            except (KeyError, TypeError):
                layer_dims = dummy_dimensions(ndims)

            dep_layer_dims[dep_layer_name] = layer_dims

            # Initialise dataset_id if not set,
            # and ensure future matches once set
            if dataset_id is None:
                dataset_id = layer_annots.get("dataset_id")
            else:
                assert dataset_id == layer_annots.get("dataset_id", dataset_id)

            # Initialise the dataset blocks if not set,
            # and ensure future matches once set
            if ds_blocks is None:
                ds_blocks = layer_annots.get("blocks")
            else:
                assert ds_blocks == layer_annots.get("blocks", ds_blocks)

        # Now, work out the dimensions of the current layer,
        # possibly from the dependent layers
        if isinstance(layer, Blockwise):
            try:
                # Defer to already existing annotations
                output_dims = layer.annotations["dims"]
            except (KeyError, TypeError):
                # Map ancestor input dimensions to output dimensions
                index_dims = dims_from_ancestors(layer, dep_layer_dims)

                # Create dummy dimensions for any new axes
                for new_axis in layer.new_axes:
                    index_dims[new_axis], = dummy_dimensions(1)

                output_dims = tuple(index_dims[i] for i in layer.output_indices)
            else:
                if len(output_dims) != len(layer.output_indices):
                    raise ValueError(f"Dimension lengths {layer_dims} != {layer.output_indices}")
        elif layer.is_materialized():
            if result := reduction_axes(layer):
                # Reduction case, the concatenation and aggregration layer
                # are a materialized graph
                # Obtain the reduction axes and keepdim property
                axis, keepdims = result
                dep_layers = hlg.dependencies[x.name]
                assert len(dep_layers) == 1, dep_layers
                dep_layer_name = next(iter(dep_layers))
                output_dims = dep_layer_dims[dep_layer_name]

                # Contraction, remove any dimensions
                # not in the axis argument
                if not keepdims:
                    output_dims = tuple(o for i, o in enumerate(output_dims) if i not in axis)
            elif result := materialized_layer_dims(layer):
                # Materialized layer
                output_dims = result
            else:
                raise NotImplementedError(f"Layer type {layer.layer_info_dict()['layer_type']}")
        else:
            raise NotImplementedError(f"Layer type {layer.layer_info_dict()['layer_type']}")

        out_annotations = {"dims": output_dims}

        if dataset_id is not None:
            out_annotations["dataset_id"] = dataset_id
        if ds_blocks is not None:
            out_annotations["blocks"] = ds_blocks

        if layer.annotations is not None:
            layer.annotations.update(out_annotations)
        else:
            layer.annotations = out_annotations

    return propagator
