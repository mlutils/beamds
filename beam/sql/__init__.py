from .core import BeamIbis
from .resource import beam_ibis
from .queries import BeamIbisQuery, TermFilter, TermsFilter, RangeFilter, TimeFilter
from .groupby import Groupby
from .schema import (
    BeamIbisSchema, 
    UserSchema, 
    EventLogSchema, 
    TimeSeriesSchema, 
    OrderSchema,
    LegacyBeamIbisSchema  # For backward compatibility
)

__all__ = [
    'BeamIbis',
    'beam_ibis', 
    'BeamIbisQuery',
    'TermFilter', 
    'TermsFilter', 
    'RangeFilter', 
    'TimeFilter',
    'Groupby',
    'BeamIbisSchema',
    'UserSchema',
    'EventLogSchema', 
    'TimeSeriesSchema',
    'OrderSchema',
    'LegacyBeamIbisSchema'
]


def __getattr__(name):
    if name == 'BeamIbis':
        from .core import BeamIbis
        return BeamIbis
    elif name == 'beam_ibis':
        from .resource import beam_ibis
        return beam_ibis
    raise AttributeError(f"module {__name__} has no attribute {name}")
