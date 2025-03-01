# pylint: disable=unused-import
from lightning.pytorch.cli import LightningCLI

from hust_bearing.models import ConvMamba
from hust_bearing.data import CWRU, HUST


def cli_main():
    LightningCLI()


if __name__ == "__main__":
    cli_main()
