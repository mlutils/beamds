# use with
# scalene --profile-all --reduced-profile --outfile import_beam_profile.html --html --- -m tests.import_beam
# import sys
# sys.path.insert(0, '../src')

import src.beam as beam


from src.beam import logger

print(beam.__version__)

print(logger.level)
