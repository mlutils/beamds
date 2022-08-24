import sys
import os


def add_beam_to_path():
    beam_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    sys.path.insert(0, beam_path)