
import datetime as _dt
import json
import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, overload

import ibis
import pandas as pd

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # noqa: N816 – signify optional dep

__all__ = [
    "BeamIbis",
    "LazySQLQuery",
]

# ---------------------------------------------------------------------------
# LazySQLQuery – records transformations and compiles lazily
# ---------------------------------------------------------------------------


class LazySQLQuery:
    """Chainable, immutable wrapper around an Ibis *Table* expression.

    Each call returns **a new object** with the updated expression recorded in
    :pyattr:`expr`.  The actual execution / SQL compilation is deferred until
    :py:meth:`execute`, :py:meth:`sql`, or dataframe accessors.
    """

    def __init__(
        self,
        backend: ibis.backends.base.BaseBackend,
        expr: ibis.expr.types.Table,
        _history: Optional[List[str]] = None,
    ) -> None:
        self._backend = backend
        self._expr = expr
        self._history: List[str] = list(_history or [])  # human‑readable log

    # ------------------------------------------------------------------
    # Transformation helpers – all return *new* LazySQLQuery instances
    # ------------------------------------------------------------------

    def filter(self, /, *predicates: Any, **field_equals: Any) -> "LazySQLQuery":
        """Filter rows via Ibis predicates OR simple field==value pairs."""
        expr = self._expr
        for pred in predicates:
            expr = expr.filter(pred)
            self._history.append(f"filter({pred})")
        for field, value in field_equals.items():
            expr = expr.filter(expr[field] == value)
            self._history.append(f"filter({field} == {value})")
        return LazySQLQuery(self._backend, expr, self._history)

    def groupby(self, *keys: str) -> "LazySQLQuery":
        self._history.append(f"groupby({keys})")
        return LazySQLQuery(self._backend, self._expr.group_by(list(keys)), self._history)

    def agg(self, **named_aggs: Tuple[str, str]) -> "LazySQLQuery":
        # named_aggs: newcol=(col, func)
        aggs = {k: getattr(self._expr[c], func)() for k, (c, func) in named_aggs.items()}
        expr = self._expr.aggregate(**aggs)
        self._history.append(f"agg({named_aggs})")
        return LazySQLQuery(self._backend, expr, self._history)

    def order_by(self, *keys: str) -> "LazySQLQuery":
        cols = []
        for k in keys:
            desc = k.startswith("-")
            c = k[1:] if desc else k
            col = self._expr[c]
            cols.append(col.desc() if desc else col)
        expr = self._expr.order_by(cols)
        self._history.append(f"order_by({keys})")
        return LazySQLQuery(self._backend, expr, self._history)

    def limit(self, n: int) -> "LazySQLQuery":
        self._history.append(f"limit({n})")
        return LazySQLQuery(self._backend, self._expr.limit(n), self._history)

    # ------------------------------------------------------------------
    # Introspection / execution
    # ------------------------------------------------------------------

    def sql(self) -> str:
        """Compile to SQL string (without executing)."""
        return self._backend.compile(self._expr)

    def execute(self, **kwargs: Any) -> pd.DataFrame:  # noqa: D401 – keep simple name
        """Run the query and return a *pandas* DataFrame."""
        return self._expr.execute(**kwargs)

    # Alias for convenience
    as_df = execute

    # ------------------------------------------------------------------
    # LLM‑powered natural‑language query helper
    # ------------------------------------------------------------------

    def ask(self, question: str, *, model: str = "gpt-4o", **chat_kwargs: Any) -> pd.DataFrame:  # noqa: D401
        """Translate *question* to SQL via an LLM and execute.

        Requires `openai` to be installed and `OPENAI_API_KEY` configured.
        The prompt exposes the *compiled schema* of the underlying table to
        the model, asks for a **single SQL statement only**, then runs it.
        """

        if openai is None:  # pragma: no cover
            raise RuntimeError("openai package not installed – `pip install openai`.")

        schema_ddl = self._expr.schema().to_summary_string()
        system = (
            "You are an expert SQL assistant. Given a BigQuery/ANSI‑SQL table "
            "schema and a user question, produce exactly ONE valid, runnable "
            "SQL query answering the question. Do not wrap it in markdown nor "
            "explain anything – return only SQL."
        )
        user = (
            f"Table schema:\n{schema_ddl}\n\n"
            f"Question: {question}\n\nSQL:"
        )

        chat = openai.ChatCompletion.create  # type: ignore[attr-defined]
        resp = chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **chat_kwargs)
        sql_code = resp.choices[0].message.content.strip().strip("`;")  # type: ignore[index]

        # Build a *new* LazySQLQuery from raw SQL and execute
        new_expr = self._backend.sql(sql_code)
        return new_expr.execute()

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover – informal repr
        return f"LazySQLQuery(sql={self.sql()!r})"

    # Enable IPython rich repr to show SQL nicely
    _repr_html_ = lambda self: f"<pre>{self.sql()}</pre>"  # noqa: E731 – small lambda ok


# ---------------------------------------------------------------------------
# BeamIbis – high‑level entry point, pathlib‑style navigation
# ---------------------------------------------------------------------------


class _DatasetAccessor(PurePosixPath):
    """Lightweight helper so `client/"dataset"` returns an accessor object."""

    def __new__(cls, root: "BeamIbis", name: str):  # noqa: D401 – required for pathlib subclass
        obj = super().__new__(cls, name)
        obj._root = root  # type: ignore[attr-defined]
        obj._name = name
        return obj

    def table(self, name: str) -> LazySQLQuery:  # noqa: D401 – simple verb
        expr = self._root._backend.table(name, database=self._name)
        return LazySQLQuery(self._root._backend, expr, [f"table({self._name}.{name})"])

    # Allow attribute access: client.dataset.table → client.dataset.table("table")

    def __getattr__(self, item: str) -> LazySQLQuery:
        return self.table(item)

    # Allow further path style: client/"dataset"/"table"

    def __truediv__(self, table: str) -> LazySQLQuery:  # type: ignore[override]
        return self.table(table)

