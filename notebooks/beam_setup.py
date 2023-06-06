# beam_setup.py
def load_ipython_extension(ipython):
    import sys
    import os

    beam_path = os.getenv('BEAM_PATH', None)
    if beam_path is not None:
        sys.path.insert(0, beam_path)

    sys.path.insert(0, '../src')
    # import beam._imports as lazy_importer

    from beam._imports import BeamImporter
    beam_importer = BeamImporter()

    # Add the modules to the global namespace
    for alias in beam_importer.aliases:
        module = getattr(beam_importer, alias)
        ipython.push({alias: module})

    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')