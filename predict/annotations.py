from functools import partial, reduce
from tlz.functoolz import Compose

import dask
from dask.highlevelgraph import HighLevelGraph
from dask.blockwise import Blockwise

from typing import Iterable, Union
from uuid import uuid4

def is_key_arraylike(key):
    return (isinstance(key, tuple) and
            isinstance(key[0], str) and
            all(isinstance(item, int) for item in key[1:]))

def reduction_axes(layer):
    try:
        k, task = next(iter(layer.items()))
    except StopIteration:
        return False

    if not is_key_arraylike(k):
        return False

    if not isinstance(task, tuple):
        return False

    fns = task[0].funcs if isinstance(task[0], Compose) else task[:1]

    try:
        pfn = next(iter(fn for fn in fns if isinstance(fn, partial)))
    except StopIteration as e:
        return False

    try:
        axis = pfn.keywords["axis"]
        keepdims = pfn.keywords["keepdims"]
    except KeyError as e:
        return False
    else:
        return axis, keepdims


def dummy_dimensions(ndims):
    return tuple(f"dummy-{uuid4().hex[:8]}" for _ in range(ndims))


def materialized_layer_dims(layer):
    try:
        k = next(iter(layer.keys()))
    except StopIteration:
        return False

    # Needs to be array like
    if not is_key_arraylike(k):
        return False

    try:
        return layer.annotations["dims"]
    except (KeyError, TypeError):
        return dummy_dimensions(len(k[1:]))

def annotate_datasets(datasets):
    ds_chunks = [dict(ds.chunks) for ds in datasets]
    ds_blocks = [{d: len(c) for d, c in chunks.items()} for chunks in ds_chunks]
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



def derive_dims_from_ancestors(layer, dep_layer_dims):
    index_dims = {}

    for dep_layer_name, indices in layer.indices:
        if indices is None:
            continue

        try:
            dep_dims = dep_layer_dims[dep_layer_name]
        except KeyError:
            # Dependent layer name might be a wierd io_dep
            try:
                layer.io_deps[dep_layer_name]
            except KeyError:
                # Should be scalar, but complain if it isn't
                if indices is not None:
                    raise ValueError(
                        f"{dep_layer_name} should be a scalar "
                        f"but indices are {indices}")
                continue

            dep_dims = dummy_dimensions(len(indices))

        if len(dep_dims) != len(indices):
            raise ValueError(f"Dimension lengths {dep_dims} != {indices}")

        index_dims.update({i: d for i, d in zip(indices, dep_dims)})

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

        # Determine the dimension schema of dependent layers
        # Make one up if one doesn't exist
        for dep_layer_name in hlg.dependencies[x.name]:
            dep_layer = hlg.layers[dep_layer_name]
            layer_annots = dep_layer.annotations or {}

            if isinstance(dep_layer, Blockwise):
                ndims = len(dep_layer.output_indices)
            elif dep_layer.is_materialized():
                k = next(iter(dep_layer.get_output_keys()))
                assert isinstance(k, tuple) and isinstance(k[0], str)
                ndims = len(k[1:])

            try:
                layer_dims = layer_annots["dims"]
            except (KeyError, TypeError):
                layer_dims = dummy_dimensions(ndims)

            dep_layer_dims[dep_layer_name] = layer_dims

            if dataset_id is None:
                dataset_id = layer_annots.get("dataset_id")
            else:
                assert dataset_id == layer_annots.get("dataset_id", dataset_id)

            if ds_blocks is None:
                ds_blocks = layer_annots.get("blocks")
            else:
                assert ds_blocks == layer_annots.get("blocks", ds_blocks)

        if isinstance(layer, Blockwise):
            # Infer output dimensions
            index_dims = {}

            try:
                # We may already have been provided with dimensions
                output_dims = layer.annotations["dims"]
            except (KeyError, TypeError):
                # Otherwise derice from dimensions provided on dependant layers
                index_dims.update(derive_dims_from_ancestors(layer, dep_layer_dims))

                for new_axis in layer.new_axes:
                    index_dims[new_axis], = dummy_dimensions(1)

                output_dims = tuple(index_dims[i] for i in layer.output_indices)
            else:
                if len(output_dims) != len(layer.output_indices):
                    raise ValueError(f"Dimension lengths {layer_dims} != {layer.output_indices}")
        elif layer.is_materialized():
            if result := reduction_axes(layer):
                # Reduction, the concatenation and aggregration layer are materialized graph
                # Get the reduction axes and keepdim property
                axis, keepdims = result
                dep_layers = hlg.dependencies[x.name]
                assert len(dep_layers) == 1, dep_layers
                dep_layer_name = next(iter(dep_layers))
                output_dims = dep_layer_dims[dep_layer_name]

                if not keepdims:
                    output_dims = tuple(o for i, o in enumerate(output_dims) if i not in axis)
            elif result := materialized_layer_dims(layer):
                output_dims = result
            else:
                raise ValueError(f"Unhandled Layer type {layer.layer_info_dict()['layer_type']}")
        else:
            raise ValueError(f"Unhandled Layer type {layer.layer_info_dict()['layer_type']}")

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
