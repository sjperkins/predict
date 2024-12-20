import logging
import math

from dask.distributed.diagnostics.plugin import SchedulerPlugin


class PinnedPlugin(SchedulerPlugin):
    def update_graph(self, scheduler, keys, annotations=None, **kwargs):
        worker_names = list(scheduler.workers.keys())
        nworkers = len(worker_names)

        try:
            dims = annotations["dims"]
            blocks = annotations["blocks"]
        except KeyError as e:
            logging.warning("PinnedPluggin found no %s "
                            "annotations and will not "
                            "pin tasks on this graph",
                            str(e))
            return

        for k, block in blocks.items():
            try:
                key_dims = dims[k]
                row_dim = key_dims.index("row")
                start, total = block["row"]
            except (KeyError, ValueError):
                continue

            assert isinstance(k[0], str)
            row_chunk = k[row_dim + 1]
            if not isinstance(row_chunk, int):
                continue

            if ts := scheduler.tasks.get(k):
                worker_id = math.floor(nworkers * ((start + row_chunk) / total))
                ts.worker_restrictions = set([worker_names[worker_id]])
                ts.loose_restrictions = True



def install_pinned_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(PinnedPlugin(**kwargs), idempotent=True)
