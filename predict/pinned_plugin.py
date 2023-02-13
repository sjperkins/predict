from ast import literal_eval
import logging
import math

from dask.distributed.diagnostics.plugin import SchedulerPlugin


class PinnedPlugin(SchedulerPlugin):
    def update_graph(self, scheduler, keys, annotations=None, **kwargs):
        worker_names = list(scheduler.workers.keys())
        #worker_names = [ws.name for ws in scheduler.workers.values()]
        nworkers = len(worker_names)

        try:
            dims = annotations["dims"]
            blocks = annotations["blocks"]
        except KeyError as e:
            logging.warning("No %s annotations could be found in PinnedPlugin", str(e))
            return

        for k, block in blocks.items():
            try:
                key_dims = dims[k]
                row_dim = key_dims.index("row")
                start, total = block["row"]
            except (KeyError, ValueError):
                continue

            # NOTE(sjperkins)
            # optimise with k.rfind(",", 0, len(k))
            tuple_key = literal_eval(k)
            row_chunk = tuple_key[row_dim + 1]
            assert isinstance(row_chunk, int)

            worker_id = math.floor(nworkers * ((start + row_chunk) / total))
            ts = scheduler.tasks.get(k)
            ts.worker_restrictions = set([worker_names[worker_id]])



def install_pinned_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(PinnedPlugin(**kwargs), idempotent=True)
