from elasticsearch_dsl.query import Query
from datetime import datetime, timedelta
import dateutil.parser
import calendar
from .dsl2kql import dsl_to_kql
from .utils import parse_kql_to_dsl


class BeamQuery(Query):
    name = 'beam_query'

    def __init__(self, query_dict=None, **kwargs):
        """
        Initialize the custom query with a query dictionary or keyword arguments.

        Args:
            query_dict (dict): A dictionary representation of the query.
            kwargs: Additional arguments to build the query.
        """
        if query_dict:
            # Pass the query dictionary to the parent Query constructor
            super().__init__(**query_dict)
        else:
            # Initialize with kwargs if no dictionary is provided
            super().__init__(**kwargs)

    @classmethod
    def from_kql(cls, kql):
        q = parse_kql_to_dsl(kql).to_dict()
        return cls(query_dict=q)


    @property
    def kql(self):
        return dsl_to_kql(self.to_dict()[self.name])



class TimeFilter(BeamQuery):
    name = "time_filter"

    def __init__(self, field=None, start=None, end=None, period=None, pattern=None, **kwargs):
        super().__init__(**kwargs)
        field_name = field or "timestamp"
        time_pattern = pattern
        start = self.parse_time(start)
        end = self.parse_time(end)
        period = self.parse_period(period)
        start, end, period = self.resolve_time_range(start, end, period)
        self._params = {"start": start, "end": end, "period": period, "field_name": field_name,
                        "time_pattern": time_pattern}

    @property
    def start(self):
        return self._params["start"]

    @property
    def end(self):
        return self._params["end"]

    @property
    def period(self):
        return self._params["period"]

    @property
    def field_name(self):
        return self._params["field_name"]

    @property
    def time_pattern(self):
        return self._params["time_pattern"]

    def parse_period(self, period):
        if isinstance(period, timedelta):
            return period
        if isinstance(period, str):
            if period.endswith("y"):
                years = int(period[:-1])
                return self.get_year_delta(years)
            elif period.endswith("m"):
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
        elif period is None:
            return None
        raise ValueError(f"Unsupported period format: {period}")

    def parse_time(self, time):
        if isinstance(time, datetime):
            return time
        elif isinstance(time, int):
            return datetime.fromtimestamp(time)
        elif isinstance(time, dict):
            now = datetime.now()
            return now.replace(**{key: time.get(key, getattr(now, key)) for key in ['year', 'month', 'day', 'hour', 'minute', 'second']})
        elif isinstance(time, str):
            if time == "now":
                return datetime.now()
            elif time == "last_year":
                now = datetime.now()
                return now.replace(year=now.year - 1)
            elif time == "last_month":
                now = datetime.now()
                if now.month == 1:
                    return now.replace(year=now.year - 1, month=12)
                return now.replace(month=now.month - 1)
            elif time == "last_day":
                return datetime.now() - timedelta(days=1)
            elif time == "last_hour":
                return datetime.now() - timedelta(hours=1)
            elif time == "last_24_hours":
                return datetime.now() - timedelta(days=1)
            if self.time_pattern:
                return datetime.strptime(time, self.time_pattern)
            return dateutil.parser.parse(time)
        elif time is None:
            return None
        raise ValueError(f"Unsupported time format: {time}")

    @staticmethod
    def get_month_delta(months):
        now = datetime.now()
        new_month = (now.month - months) % 12 or 12
        new_year = now.year + (now.month - months - 1) // 12
        last_day = calendar.monthrange(new_year, new_month)[1]
        return now - now.replace(year=new_year, month=new_month, day=min(now.day, last_day))

    @staticmethod
    def get_year_delta(years):
        now = datetime.now()
        try:
            return now - now.replace(year=now.year - years)
        except ValueError:  # Handle leap years
            return now - now.replace(year=now.year - years, day=28)

    @staticmethod
    def resolve_time_range(start, end, period):
        if start is None and end is not None and period is not None:
            start = end - period
        elif end is None and start is not None and period is not None:
            end = start + period
        elif end is None and period is not None and start is None:
            end = datetime.now()
            start = end - period
        return start, end, period

    def _clone(self):
        c = self.__class__(field=self.field_name, start=self.start, end=self.end, period=self.period,
                           pattern=self.time_pattern)
        return c

    def to_dict(self):
        d = {"range": {self.field_name: dict()}}

        if self.start is not None:
            d["range"][self.field_name]["gte"] = self.start.isoformat()
        if self.end is not None:
            d["range"][self.field_name]["lte"] = self.end.isoformat()

        return d

    def __str__(self):
        # use timeformat to print the time in a human readable format, if format is not provided,
        # use elasticsearch format (with time zone)
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
        return f"TimeFilter:({str(self)})"


