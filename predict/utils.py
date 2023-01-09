from collections.abc import Iterable
import toolz


def _fuse_annotations(*args: dict) -> dict:
    """
    Given an iterable of annotations dictionaries, fuse them according
    to some simple rules.
    """
    # First, do a basic dict merge -- we are presuming that these have already
    # been gated by `_can_fuse_annotations`.
    annotations = toolz.merge(*args)
    # Max of layer retries
    retries = [a["retries"] for a in args if "retries" in a]
    if retries:
        annotations["retries"] = max(retries)
    # Max of layer priorities
    priorities = [a["priority"] for a in args if "priority" in a]
    if priorities:
        annotations["priority"] = max(priorities)
    # Max of all the layer resources
    resources = [a["resources"] for a in args if "resources" in a]
    if resources:
        annotations["resources"] = toolz.merge_with(max, *resources)
    # Intersection of all the worker restrictions
    workers = [a["workers"] for a in args if "workers" in a]
    if workers:
        annotations["workers"] = list(
            set.intersection(
                *[set(w) if isinstance(w, Iterable) else set([w]) for w in workers]
            )
        )
    # More restrictive of allow_other_workers
    allow_other_workers = [
        a["allow_other_workers"] for a in args if "allow_other_workers" in a
    ]
    if allow_other_workers:
        annotations["allow_other_workers"] = all(allow_other_workers)

    return annotations
