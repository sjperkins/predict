import argparse
import logging
import sys
from contextlib import ExitStack
from typing import Dict, Iterable
from unittest import mock

from dask.distributed import Client, LocalCluster

from predict.prediction import predict_vis
from predict.sky_model import generate_sky_model
from predict.utils import _fuse_annotations


class Application:
    DEFAULT_CHANS = 4096
    DEFAULT_SOURCES = 1000

    DEFAULT_ROW_CHUNKS = 10000
    DEFAULT_CHAN_CHUNKS = 64
    DEFAULT_SOURCE_CHUNKS = 100

    DEFAULT_DIMENSIONS = f"{{chan: {DEFAULT_CHANS}," f"source: {DEFAULT_SOURCES}}}"

    DEFAULT_CHUNKS = (
        f"{{row: {DEFAULT_ROW_CHUNKS},"
        f"chan: {DEFAULT_CHAN_CHUNKS},"
        f"source: {DEFAULT_SOURCE_CHUNKS}}}"
    )

    def __init__(self, args: Iterable[str]):
        self.args = self.parse_args(args)

    @staticmethod
    def parse_dim_dict(dims: str) -> Dict[str, int]:
        msg = f"'{dims}' does not conform to {{d1: s1, d2: s2, ..., dn: sn}}"
        err = argparse.ArgumentTypeError(msg)
        dims = dims.strip()

        if not dims[0] == "{" and dims[-1] == "}":
            raise err

        result = {}

        for dim in [d.strip() for d in dims[1:-1].split(",")]:
            bits = [b.strip() for b in dim.split(":")]

            if len(bits) != 2:
                raise err

            try:
                result[bits[0]] = int(bits[1])
            except ValueError:
                raise err

        return result

    @staticmethod
    def parse_args(args: Iterable[str]) -> argparse.Namespace:
        p = argparse.ArgumentParser()
        p.add_argument("store", help="Measurement Set store")
        p.add_argument("--output-store", required=False)
        p.add_argument("--output-column", default="MODEL_DATA")
        p.add_argument("--address", help="distributed scheduler address")
        p.add_argument(
            "--workers", help="number of distributed workers", type=int, default=8
        )
        p.add_argument(
            "--dimensions",
            default=Application.DEFAULT_DIMENSIONS,
            type=Application.parse_dim_dict,
        )

        p.add_argument(
            "--chunks",
            default=Application.DEFAULT_CHUNKS,
            type=Application.parse_dim_dict,
        )

        args = p.parse_args(args)

        if not args.output_store:
            args.output_store = args.store

        args.dimensions.setdefault("chan", Application.DEFAULT_CHANS)
        args.dimensions.setdefault("source", Application.DEFAULT_SOURCES)

        args.chunks.setdefault("row", Application.DEFAULT_ROW_CHUNKS)
        args.chunks.setdefault("chan", Application.DEFAULT_CHAN_CHUNKS)
        args.chunks.setdefault("source", Application.DEFAULT_SOURCE_CHUNKS)

        return args

    @staticmethod
    def get_client(args: argparse.Namespace, stack: ExitStack) -> Client:
        if not args.address:
            cluster = stack.enter_context(
                LocalCluster(
                    n_workers=args.workers, processes=False, threads_per_worker=1
                )
            )
            address = cluster.scheduler_address
            logging.info("Created LocalCluster with Scheduler at %s", address)

        else:
            address = args.address
            logging.info("Connected to Distributed Scheduler at %s", address)

        return stack.enter_context(Client(address))

    def run(self):
        logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch("dask.blockwise._fuse_annotations", _fuse_annotations)
            )
            client = self.get_client(self.args, stack)
            logging.info("Waiting for %d workers to be ready", self.args.workers)
            client.wait_for_workers(self.args.workers)
            client.amm.stop()
            logging.info(
                "Generating sky model of %s sources", self.args.dimensions["source"]
            )
            model = generate_sky_model(self.args)

            logging.info("Predicting Visibilities")
            predict_vis(self.args, model)
            logging.info("Done")


def main():
    Application(sys.argv[1:]).run()


if __name__ == "__main__":
    main()
