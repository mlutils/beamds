
# ------------------------------------------------------------------
# Aggregations & groupby
# ------------------------------------------------------------------
class GroupByHelper:
    """Thin wrapper around ibis.GroupedTable to mimic Elastic's Groupby helper."""

    def __init__(self, parent: "BeamBigQuery", fields: list[str]):
        self.parent = parent
        self.fields = fields
        self._metrics: dict[str, tuple[str, str]] = {}
        # mapping from alias -> (op, field)

    # Metric builders
    def _add(self, alias: str, op: str, field: str):
        self._metrics[alias] = (op, field)
        return self

    def sum(self, field):
        return self._add(f"{field}_sum", "sum", field)

    def mean(self, field):
        return self._add(f"{field}_mean", "mean", field)

    def min(self, field):
        return self._add(f"{field}_min", "min", field)

    def max(self, field):
        return self._add(f"{field}_max", "max", field)

    def nunique(self, field):
        return self._add(f"{field}_nunique", "nunique", field)

    def count(self, field="*"):
        return self._add("count", "count", field)

    # Execute
    def _apply(self):
        base = self.parent._current_expr()
        grouped = base.group_by(self.fields)
        metrics = {}
        for alias, (op, field) in self._metrics.items():
            col = ibis.field(field) if field != "*" else None
            match op:
                case "sum":
                    metrics[alias] = col.sum()
                case "mean":
                    metrics[alias] = col.mean()
                case "min":
                    metrics[alias] = col.min()
                case "max":
                    metrics[alias] = col.max()
                case "nunique":
                    metrics[alias] = col.nunique()
                case "count":
                    metrics[alias] = ibis.literal(1).count() if col is None else col.count()
                case _:
                    raise ValueError(f"Unknown agg op {op}")
        result_expr = grouped.aggregate(**metrics)
        return result_expr

    # Materialisers
    def as_df(self):
        return self._apply().execute(limit=None)

    def as_pl(self):
        import polars as pl
        return pl.from_pandas(self.as_df())

    def as_cudf(self):
        import cudf
        return cudf.from_pandas(self.as_df())
