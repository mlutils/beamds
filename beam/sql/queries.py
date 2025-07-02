import datetime as _dt
from datetime import datetime, timedelta
import calendar


def _now():
    return _dt.datetime.now(tz=_dt.timezone.utc)


class BeamIbisQuery:
    """Base query class for Ibis expressions."""
    
    def __init__(self, expr=None, **kwargs):
        """
        Initialize the custom query with an Ibis expression.

        Args:
            expr: An Ibis expression
            kwargs: Additional arguments to build the query.
        """
        self.expr = expr
        self.kwargs = kwargs

    def to_expr(self):
        """Convert to Ibis expression."""
        return self.expr

    def __and__(self, other):
        """Combine with another query using AND logic."""
        if hasattr(other, 'to_expr'):
            other_expr = other.to_expr()
        else:
            other_expr = other
        
        if self.expr is not None and other_expr is not None:
            return BeamIbisQuery(self.expr & other_expr)
        return BeamIbisQuery(other_expr or self.expr)

    def __or__(self, other):
        """Combine with another query using OR logic."""
        if hasattr(other, 'to_expr'):
            other_expr = other.to_expr()
        else:
            other_expr = other
        
        if self.expr is not None and other_expr is not None:
            return BeamIbisQuery(self.expr | other_expr)
        return BeamIbisQuery(other_expr or self.expr)

    def __invert__(self):
        """Negate the query."""
        if self.expr is not None:
            return BeamIbisQuery(~self.expr)
        return self

    def __str__(self):
        return str(self.expr) if self.expr is not None else "EmptyQuery"

    def __repr__(self):
        return f"BeamIbisQuery({self.expr})"


class TimeFilter(BeamIbisQuery):
    """Time filter for Ibis queries, similar to BeamElastic TimeFilter."""
    
    def __init__(self, backend=None, table=None, field=None, start=None, end=None, period=None, pattern=None, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.table = table
        self.field_name = field or "timestamp"
        self.time_pattern = pattern
        self.start = self.parse_time(start)
        self.end = self.parse_time(end)
        self.period = self.parse_period(period)
        self.start, self.end, self.period = self.resolve_time_range(self.start, self.end, self.period)

    def parse_period(self, period):
        """Parse period string or timedelta."""
        if isinstance(period, timedelta):
            return period
        if isinstance(period, str):
            if period.endswith("y"):
                years = int(period[:-1])
                return self.get_year_delta(years)
            elif period.endswith("M"):  # Month
                months = int(period[:-1])
                return self.get_month_delta(months)
            elif period.endswith("d"):
                return timedelta(days=int(period[:-1]))
            elif period.endswith("h"):
                return timedelta(hours=int(period[:-1]))
            elif period.endswith("s"):
                return timedelta(seconds=int(period[:-1]))
            elif period.endswith("min"):
                return timedelta(minutes=int(period[:-3]))
            elif period.endswith("w"):
                return timedelta(weeks=int(period[:-1]))
        elif period is None:
            return None
        raise ValueError(f"Unsupported period format: {period}")

    def parse_time(self, time):
        """Parse various time formats."""
        if isinstance(time, datetime):
            return time
        elif isinstance(time, int):
            return datetime.fromtimestamp(time)
        elif isinstance(time, dict):
            now = datetime.now()
            return now.replace(**{key: time.get(key, getattr(now, key)) for key in ['year', 'month', 'day', 'hour', 'minute', 'second']})
        elif isinstance(time, str):
            if time == "now":
                return _now()
            elif time == "last_year":
                now = _now()
                return now.replace(year=now.year - 1)
            elif time == "last_month":
                now = _now()
                if now.month == 1:
                    return now.replace(year=now.year - 1, month=12)
                return now.replace(month=now.month - 1)
            elif time == "last_day":
                return _now() - timedelta(days=1)
            elif time == "last_hour":
                return _now() - timedelta(hours=1)
            elif time == "last_24_hours":
                return _now() - timedelta(days=1)
            elif time == "last_week":
                return _now() - timedelta(weeks=1)
            
            # Try to parse as ISO format or custom pattern
            if self.time_pattern:
                return datetime.strptime(time, self.time_pattern)
            else:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S%z']:
                    try:
                        return datetime.strptime(time, fmt)
                    except ValueError:
                        continue
                # If all else fails, try dateutil
                try:
                    import dateutil.parser
                    return dateutil.parser.parse(time)
                except:
                    raise ValueError(f"Unable to parse time string: {time}")
        elif time is None:
            return None
        raise ValueError(f"Unsupported time format: {time}")

    @staticmethod
    def get_month_delta(months):
        """Get timedelta for months (approximation)."""
        now = _now()
        try:
            new_month = (now.month - months) % 12 or 12
            new_year = now.year + (now.month - months - 1) // 12
            last_day = calendar.monthrange(new_year, new_month)[1]
            past_date = now.replace(year=new_year, month=new_month, day=min(now.day, last_day))
            return now - past_date
        except ValueError:
            # Fallback to approximation
            return timedelta(days=months * 30)

    @staticmethod
    def get_year_delta(years):
        """Get timedelta for years."""
        now = _now()
        try:
            return now - now.replace(year=now.year - years)
        except ValueError:  # Handle leap years
            return now - now.replace(year=now.year - years, day=28)

    @staticmethod
    def resolve_time_range(start, end, period):
        """Resolve start/end times from period."""
        if start is None and end is not None and period is not None:
            start = end - period
        elif end is None and start is not None and period is not None:
            end = start + period
        elif end is None and period is not None and start is None:
            end = _now()
            start = end - period
        return start, end, period

    def to_expr(self):
        """Convert to Ibis expression."""
        if self.table is None:
            raise ValueError("Table is required for time filter")
        
        predicates = []
        
        if self.start is not None:
            predicates.append(self.table[self.field_name] >= self.start)
        if self.end is not None:
            predicates.append(self.table[self.field_name] <= self.end)
        
        if not predicates:
            return None
        
        # Combine predicates with AND
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        
        return result

    def apply_to_table(self, table):
        """Apply the time filter to a table."""
        expr = self.to_expr()
        if expr is not None:
            return table.filter(expr)
        return table

    def __str__(self):
        """Human-readable representation."""
        pattern = self.time_pattern or "%Y-%m-%d %H:%M:%S"
        start = None
        if self.start:
            start = self.start.strftime(pattern)
        end = None
        if self.end:
            end = self.end.strftime(pattern)

        if start and end:
            return f"{self.field_name} between {start} and {end}"
        if start:
            return f"{self.field_name} after {start}"
        if end:
            return f"{self.field_name} before {end}"
        return f"{self.field_name} not set"

    def __repr__(self):
        return f"TimeFilter({str(self)})"


class RangeFilter(BeamIbisQuery):
    """Range filter for numeric fields."""
    
    def __init__(self, table=None, field=None, min_val=None, max_val=None, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.field_name = field
        self.min_val = min_val
        self.max_val = max_val

    def to_expr(self):
        """Convert to Ibis expression."""
        if self.table is None:
            raise ValueError("Table is required for range filter")
        
        predicates = []
        
        if self.min_val is not None:
            predicates.append(self.table[self.field_name] >= self.min_val)
        if self.max_val is not None:
            predicates.append(self.table[self.field_name] <= self.max_val)
        
        if not predicates:
            return None
        
        # Combine predicates with AND
        result = predicates[0]
        for pred in predicates[1:]:
            result = result & pred
        
        return result

    def __str__(self):
        if self.min_val is not None and self.max_val is not None:
            return f"{self.field_name} between {self.min_val} and {self.max_val}"
        elif self.min_val is not None:
            return f"{self.field_name} >= {self.min_val}"
        elif self.max_val is not None:
            return f"{self.field_name} <= {self.max_val}"
        return f"{self.field_name} range filter"

    def __repr__(self):
        return f"RangeFilter({str(self)})"


class TermFilter(BeamIbisQuery):
    """Term filter for exact matches."""
    
    def __init__(self, table=None, field=None, value=None, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.field_name = field
        self.value = value

    def to_expr(self):
        """Convert to Ibis expression."""
        if self.table is None:
            raise ValueError("Table is required for term filter")
        
        return self.table[self.field_name] == self.value

    def __str__(self):
        return f"{self.field_name} == {self.value}"

    def __repr__(self):
        return f"TermFilter({str(self)})"


class TermsFilter(BeamIbisQuery):
    """Terms filter for multiple values (IN clause)."""
    
    def __init__(self, table=None, field=None, values=None, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.field_name = field
        self.values = values or []

    def to_expr(self):
        """Convert to Ibis expression."""
        if self.table is None:
            raise ValueError("Table is required for terms filter")
        
        return self.table[self.field_name].isin(self.values)

    def __str__(self):
        return f"{self.field_name} in {self.values}"

    def __repr__(self):
        return f"TermsFilter({str(self)})" 