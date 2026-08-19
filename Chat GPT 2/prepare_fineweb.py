"""
FineWeb-Edu dataset preparation.

Run:
$ python prepare_fineweb.py

This saves GPT-2-tokenized FineWeb-Edu shards to data/edu_fineweb10B/.
"""

from datautils import prepare_fineweb


if __name__ == "__main__":
    prepare_fineweb()
