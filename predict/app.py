import argparse
from contextlib import ExitStack
import sys

from distributed import Client, LocalCluster

from typing import Dict, List

from predict.sky_model import load_sky_model
from predict.prediction import predict_vis


class Application:
    DEFAULT_ROWS = 10000
    DEFAULT_CHANS = 64
    DEFAULT_SOURCES = 100

    DEFAULT_CHUNKS = (
        f"{{row: {DEFAULT_ROWS}, chan: {DEFAULT_CHANS}, source: {DEFAULT_SOURCES}}}"
    )

    def __init__(self, args: List[str]):
        self.args = self.parse_args(args)

    @staticmethod
    def parse_chunks(chunks: str) -> Dict[str, int]:
        msg = f"'{chunks}' does not conform to {{d1: s1, d2: s2, ..., dn: sn}}"
        err = argparse.ArgumentTypeError(msg)
        chunks = chunks.strip()

        if not chunks[0] == "{" and chunks[-1] == "}":
            raise err

        result = {}

        for chunk in [c.strip() for c in chunks[1:-1].split(",")]:
            bits = [b.strip() for b in chunk.split(":")]

            if len(bits) != 2:
                raise err

            try:
                result[bits[0]] = int(bits[1])
            except ValueError:
                raise err

        result.setdefault("row", Application.DEFAULT_ROWS)
        result.setdefault("chan", Application.DEFAULT_CHANS)
        result.setdefault("source", Application.DEFAULT_SOURCES)

        return result

    @staticmethod
    def parse_args(args: List[str]) -> argparse.Namespace:
        p = argparse.ArgumentParser()
        p.add_argument("store", help="Measurement Set store")
        p.add_argument("--sky-model", help="Sky model file", required=True)
        p.add_argument("--output-store", required=False)
        p.add_argument("--output-column", default="MODEL_DATA")
        p.add_argument("--address", help="distributed scheduler address")
        p.add_argument(
            "--workers", help="number of distributed workers", type=int, default=8
        )
        p.add_argument(
            "--chunks",
            default=Application.DEFAULT_CHUNKS,
            type=Application.parse_chunks,
        )

        args = p.parse_args(args)

        if not args.output_store:
            args.output_store = args.store

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
        else:
            address = args.address

        return stack.enter_context(Client(address))

    def run(self):
        with ExitStack() as stack:
            client = self.get_client(self.args, stack)
            model = load_sky_model(self.args.sky_model)

            predict_vis(self.args, model)


def main():
    Application(sys.argv[1:]).run()

if __name__ == "__main__":
    main()
