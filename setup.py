from setuptools import setup


setup(
    name="hust-bearing",
    packages=["hust_bearing"],
    entry_points={
        "console_scripts": [
            "hust-bearing = hust_bearing.main:cli_main",
        ],
    },
)
