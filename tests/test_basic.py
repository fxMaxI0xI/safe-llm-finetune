import importlib
import argparse


def test_import_pipeline():
    assert importlib.import_module("pipeline")
