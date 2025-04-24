from __future__ import annotations

"""airflow_client.py
====================

A *filesystem‑like* wrapper around **apache‑airflow‑client‑python** that exposes
canonical Airflow lifecycle operations at four logical levels while fully
respecting query‑driven *filtered* views.

Levels & filters
----------------
* ``root`` / ``filtered_dags``        – list DAGs (filters: ``only_active``, ``paused``, ``dag_id_pattern``)
* ``dag`` / ``filtered_dag_runs``     – list DAG‑runs (filters: ``state`` + date range)
* ``dag_run`` / ``filtered_tasks``    – list task‑instances (filters: ``state``)
* ``task_instance``                   – a single task instance

Each canonical operation acts only on the objects visible at the current
level (or raises if meaningless).  For example ``clear()`` on
``filtered_dag_runs`` will clear every DAG‑run matching the query, whereas
``clear()`` on ``dag_run`` clears just that one run.
"""

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union

from airflow_client.client import ApiClient, Configuration
from airflow_client.client.api.dag_api import DAGApi
from airflow_client.client.api.dag_run_api import DAGRunApi
from airflow_client.client.api.task_instance_api import TaskInstanceApi
from airflow_client.client.api.config_api import ConfigApi
from airflow_client.client.model.dag_run import DAGRun
from airflow_client.client.model.clear_task_instances import ClearTaskInstances

# project‑local utilities (path‑like behaviour & host normalisation)
from ..path import PureBeamPath, normalize_host

__all__ = [
    "AirflowQuery",
    "AirflowClient",
]


###############################################################################
# Query helper
###############################################################################

class AirflowQuery:
    """Thin container for list‑endpoint filters & ordering."""

    def __init__(
        self,
        *,
        state: Optional[Union[str, List[str]]] = None,
        execution_date_gte: Optional[datetime] = None,
        execution_date_lte: Optional[datetime] = None,
        start_date_gte: Optional[datetime] = None,
        start_date_lte: Optional[datetime] = None,
        end_date_gte: Optional[datetime] = None,
        end_date_lte: Optional[datetime] = None,
        order_by: Optional[str] = None,
        # DAG‑list specific filters
        only_active: Optional[bool] = None,
        paused: Optional[bool] = None,
        dag_id_pattern: Optional[str] = None,
    ) -> None:
        self.state = list(state) if isinstance(state, (list, tuple)) else ([state] if state else None)
        self.execution_date_gte = execution_date_gte
        self.execution_date_lte = execution_date_lte
        self.start_date_gte = start_date_gte
        self.start_date_lte = start_date_lte
        self.end_date_gte = end_date_gte
        self.end_date_lte = end_date_lte
        self.order_by = order_by
        self.only_active = only_active
        self.paused = paused
        self.dag_id_pattern = dag_id_pattern

    # ------------------------------------------------------------------
    # serialisation helpers
    # ------------------------------------------------------------------

    def to_params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        if self.state:
            p["state"] = self.state
        if self.execution_date_gte:
            p["execution_date_gte"] = self.execution_date_gte
        if self.execution_date_lte:
            p["execution_date_lte"] = self.execution_date_lte
        if self.start_date_gte:
            p["start_date_gte"] = self.start_date_gte
        if self.start_date_lte:
            p["start_date_lte"] = self.start_date_lte
        if self.end_date_gte:
            p["end_date_gte"] = self.end_date_gte
        if self.end_date_lte:
            p["end_date_lte"] = self.end_date_lte
        if self.order_by:
            p["order_by"] = self.order_by
        if self.only_active is not None:
            p["only_active"] = self.only_active
        if self.paused is not None:
            p["paused"] = self.paused
        if self.dag_id_pattern:
            p["dag_id_pattern"] = self.dag_id_pattern
        return p

    # ------------------------------------------------------------------
    # boolean algebra ("&" = intersection, "|" = union)
    # ------------------------------------------------------------------

    def _merge(self, a, b, fn):
        return fn(filter(None, [a, b])) if any([a, b]) else None

    def __and__(self, other: "AirflowQuery") -> "AirflowQuery":
        if not isinstance(other, AirflowQuery):
            raise TypeError("Operands must be AirflowQuery instances")
        return AirflowQuery(
            state=list(set(self.state or []).intersection(set(other.state or []))) or None,
            execution_date_gte=self._merge(self.execution_date_gte, other.execution_date_gte, max),
            execution_date_lte=self._merge(self.execution_date_lte, other.execution_date_lte, min),
            start_date_gte=self._merge(self.start_date_gte, other.start_date_gte, max),
            start_date_lte=self._merge(self.start_date_lte, other.start_date_lte, min),
            end_date_gte=self._merge(self.end_date_gte, other.end_date_gte, max),
            end_date_lte=self._merge(self.end_date_lte, other.end_date_lte, min),
            order_by=self.order_by or other.order_by,
            only_active=self.only_active if self.only_active is not None else other.only_active,
            paused=self.paused if self.paused is not None else other.paused,
            dag_id_pattern=self.dag_id_pattern or other.dag_id_pattern,
        )

    def __or__(self, other: "AirflowQuery") -> "AirflowQuery":
        if not isinstance(other, AirflowQuery):
            raise TypeError("Operands must be AirflowQuery instances")
        return AirflowQuery(
            state=list(set(self.state or []).union(set(other.state or []))) or None,
            execution_date_gte=self._merge(self.execution_date_gte, other.execution_date_gte, min),
            execution_date_lte=self._merge(self.execution_date_lte, other.execution_date_lte, max),
            start_date_gte=self._merge(self.start_date_gte, other.start_date_gte, min),
            start_date_lte=self._merge(self.start_date_lte, other.start_date_lte, max),
            end_date_gte=self._merge(self.end_date_gte, other.end_date_gte, min),
            end_date_lte=self._merge(self.end_date_lte, other.end_date_lte, max),
            order_by=self.order_by or other.order_by,
            only_active=self.only_active if self.only_active is not None else other.only_active,
            paused=self.paused if self.paused is not None else other.paused,
            dag_id_pattern=self.dag_id_pattern or other.dag_id_pattern,
        )

    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        return ", ".join(f"{k}={v}" for k, v in self.to_params().items())

    @classmethod
    def parser(cls, q: Union[None, "AirflowQuery", Dict[str, Any]]) -> Optional["AirflowQuery"]:
        if isinstance(q, cls):
            return q
        if isinstance(q, dict):
            return cls(**q)
        return None


###############################################################################
# Main client
###############################################################################

class AirflowClient(PureBeamPath):
    """Filesystem‑like Airflow client exposing lifecycle operations."""

    PAGE_SIZE = 100

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        *pathsegments: str,
        client: Optional[ApiClient] = None,
        hostname: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: Union[bool, str] = False,
        verify: bool = False,
        q: Union[None, AirflowQuery, Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not pathsegments:
            pathsegments = ("/",)
        super().__init__(*pathsegments, scheme="airflow", client=client, hostname=hostname, port=port,
                         username=username, password=password, tls=tls, verify=verify, **kwargs)

        proto = "https" if (isinstance(tls, str) and tls.lower() == "true") or tls is True else "http"
        self._url = f"{proto}://{normalize_host(hostname, port)}/api/v1"

        if client is None:
            cfg = Configuration(host=self._url, username=username, password=password)
            cfg.verify_ssl = verify
            client = ApiClient(cfg)
        self.client: ApiClient = client

        # shortcut APIs
        self.dag_api = DAGApi(client)
        self.dag_run_api = DAGRunApi(client)
        self.task_api = TaskInstanceApi(client)
        self.config_api = ConfigApi(client)

        self._q: Optional[AirflowQuery] = AirflowQuery.parser(q)

    # ------------------------------------------------------------------
    # level helpers & path parts
    # ------------------------------------------------------------------

    @property
    def level(self) -> str:
        depth = len(self.parts[1:])
        if self._q is None:
            return {0: "root", 1: "dag", 2: "dag_run", 3: "task_instance"}[depth]
        return {0: "filtered_dags", 1: "filtered_dag_runs", 2: "filtered_tasks", 3: "task_instance"}[depth]

    @property
    def dag_id(self) -> Optional[str]:
        return self.parts[1] if len(self.parts) > 1 else None

    @property
    def run_id(self) -> Optional[str]:
        return self.parts[2] if len(self.parts) > 2 else None

    @property
    def task_id(self) -> Optional[str]:
        return self.parts[3] if len(self.parts) > 3 else None

    # ------------------------------------------------------------------
    # health & utility
    # ------------------------------------------------------------------

    def health(self) -> str:
        try:
            res = self.client.rest_client.GET(f"{self._url}/health")
            return json.loads(res.data)["metadatabase"]["status"]
        except Exception as exc:  # pragma: no cover
            return str(exc)

    def ping(self) -> bool:  # pragma: no cover
        return self.health() == "healthy"

    # ------------------------------------------------------------------
    # path factory preserving connection info
    # ------------------------------------------------------------------

    def gen(self, path: str, **kwargs) -> "AirflowClient":
        base = dict(hostname=self.hostname, port=self.port, username=self.username, password=self.password,
                    q=self._q, fragment=self.fragment, params=self.params)
        base.update(kwargs)
        return type(self)(path, client=self.client, **base)

    # ------------------------------------------------------------------
    # directory iteration (list DAGs / runs / tasks)
    # ------------------------------------------------------------------

    def iterdir(self, query: Optional[AirflowQuery] = None) -> Iterable["AirflowClient"]:
        query = query or self._q
        qp = query.to_params() if query else {}

        if self.level in ("root", "filtered_dags"):
            filters = {k: qp.get(k) for k in ("only_active", "paused", "dag_id_pattern") if k in qp}
            for d in self.dag_api.get_dags(limit=self.PAGE_SIZE, **filters).dags:
                yield self.joinpath(d.dag_id)

        elif self.level in ("dag", "filtered_dag_runs"):
            offset = 0
            while True:
                page = self.dag_run_api.get_dag_runs(self.dag_id, limit=self.PAGE_SIZE, offset=offset, **qp)
                for dr in page.dag_runs:
                    yield self.joinpath(dr.dag_run_id)
                if len(page.dag_runs) < self.PAGE_SIZE:
                    break
                offset += self.PAGE_SIZE

        elif self.level in ("dag_run", "filtered_tasks"):
            for ti in self.task_api.get_task_instances(self.dag_id, self.run_id, **qp):
                yield self.joinpath(ti.task_id)
        else:
            raise ValueError("Cannot iterate below task_instance level")

    # ------------------------------------------------------------------
    # helper generators for bulk ops when filters are active
    # ------------------------------------------------------------------

    def _iter_dag_runs(self, qp) -> Iterable[str]:
        offset = 0
        while True:
            page = self.dag_run_api.get_dag_runs(self.dag_id, limit=self.PAGE_SIZE, offset=offset, **qp)
            for dr in page.dag_runs:
                yield dr.dag_run_id
            if len(page.dag_runs) < self.PAGE_SIZE:
                break
            offset += self.PAGE_SIZE

    def _iter_task_instances(self, qp) -> Iterable[str]:
        for ti in self.task_api.get_task_instances(self.dag_id, self.run_id, **qp):
            yield ti.task_id

    # ------------------------------------------------------------------
    # Canonical lifecycle operations (filtered‑aware)
    # ------------------------------------------------------------------

    # ---- trigger ------------------------------------------------------

    def trigger(self, *, conf: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> None:
        """Trigger execution depending on level.

        * dag            – create new DAG‑run
        * dag_run        – clear all tasks (rerun)
        * task_instance  – clear just this task (rerun)
        """
        if self.level == "dag":
            rid = run_id or f"manual__{datetime.utcnow().isoformat()}"
            self.dag_run_api.post_dag_run(self.dag_id, DAGRun(run_id=rid, conf=conf))
        elif self.level == "dag_run":
            self.clear()  # clear all tasks in the run, scheduler will pick them up
        elif self.level == "task_instance":
            self.task_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, ClearTaskInstances())
        else:
            raise ValueError("trigger not supported at this path level")

    # ---- clear --------------------------------------------------------

    def clear(self, tasks: Optional[List[str]] = None) -> None:
        qp = self._q.to_params() if self._q else {}
        if self.level == "dag_run":
            body = ClearTaskInstances(task_ids=tasks) if tasks else ClearTaskInstances()
            self.dag_run_api.clear_dag_run(self.dag_id, self.run_id, clear_task_instances=body)
        elif self.level == "task_instance":
            self.task_api.clear_task_instance(self.dag_id, self.run_id, self.task_id, ClearTaskInstances())
        elif self.level == "filtered_dag_runs":
            for rid in self._iter_dag_runs(qp):
                self.dag_run_api.clear_dag_run(self.dag_id, rid, clear_task_instances=ClearTaskInstances())
        elif self.level == "filtered_tasks":
            for tid in self._iter_task_instances(qp):
                self.task_api.clear_task_instance(self.dag_id, self.run_id, tid, ClearTaskInstances())
        else:
            raise ValueError("clear not valid at this level")

    # ---- mark_success / mark_failed -----------------------------------

    def _update_state(self, new_state: str) -> None:
        qp = self._q.to_params() if self._q else {}
        if self.level == "dag_run":
            self.dag_run_api.update_dag_run_state(self.dag_id, self.run_id, state=new_state)
        elif self.level == "task_instance":
            self.task_api.update_task_instance_state(self.dag_id, self.run_id, self.task_id, state=new_state)
        elif self.level == "filtered_dag_runs":
            for rid in self._iter_dag_runs(qp):
                self.dag_run_api.update_dag_run_state(self.dag_id, rid, state=new_state)
        elif self.level == "filtered_tasks":
            for tid in self._iter_task_instances(qp):
                self.task_api.update_task_instance_state(self.dag_id, self.run_id, tid, state=new_state)
        else:
            raise ValueError(f"mark_{new_state} not supported at this level")

    def mark_success(self) -> None:
        self._update_state("success")

    def mark_failed(self) -> None:
        self._update_state("failed")

    # ---- delete -------------------------------------------------------

    def delete(self) -> None:
        qp = self._q.to_params() if self._q else {}
        if self.level == "dag":
            self.dag_api.delete_dag(self.dag_id)
        elif self.level == "dag_run":
            self.dag_run_api.delete_dag_run(self.dag_id, self.run_id)
        elif self.level == "filtered_dag_runs":
            for rid in self._iter_dag_runs(qp):
                self.dag_run_api.delete_dag_run(self.dag_id, rid)
        else:
            raise ValueError("delete not supported at this path level")

    # ---- pause / unpause / backfill -----------------------------------

    def pause(self) -> None:
        if self.level != "dag":
            raise ValueError("pause only valid at DAG level")
        self.dag_api.update_dag(self.dag_id, {"is_paused": True})

    def unpause(self) -> None:
        if self.level != "dag":
            raise ValueError("unpause only valid at DAG level")
        self.dag_api.update_dag(self.dag_id, {"is_paused": False})

    def backfill(self, *, start_date: datetime, end_date: datetime, conf: Optional[Dict[str, Any]] = None) -> None:
        if self.level != "dag":
            raise ValueError("backfill only makes sense at DAG level")
        self.dag_api.post_dag_backfill(self.dag_id, {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "conf": conf or {},
        })

    # ---- run / logs (task test) ---------------------------------------

    def run(self) -> None:
        if self.level != "task_instance":
            raise ValueError("run() valid only at task_instance level")
        self.task_api.post_task_instance(self.dag_id, self.run_id, self.task_id)

    def logs(self, try_number: int = 1) -> str:
        if self.level != "task_instance":
            raise ValueError("logs() valid only at task_instance level")
        return self.task_api.get_log(self.dag_id, self.run_id, self.task_id, task_try_number=try_number)

    # ------------------------------------------------------------------
    # convenience wrappers for filtered listings
    # ------------------------------------------------------------------

    def iter_success(self, **kwargs):
        return self.iterdir(AirflowQuery(state="success", **kwargs))

    def iter_failed(self, **kwargs):
        return self.iterdir(AirflowQuery(state="failed", **kwargs))

    def iter_running(self, **kwargs):
        return self.iterdir(AirflowQuery(state="running", **kwargs))

    # ------------------------------------------------------------------
    # rich representation & boolean AND (carry filters)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        out = str(self.url)
        if self._q:
            q = str(self._q)
            out += f" | query: {q[:60]}…" if len(q) > 60 else f" | query: {q}"
        return out

    def __and__(self, other: Union["AirflowClient", AirflowQuery, Dict[str, Any]]) -> "AirflowClient":
        new_q: Optional[AirflowQuery]
        if isinstance(other, AirflowClient):
            new_q = other._q
        else:
            new_q = AirflowQuery.parser(other)
        combined = self._q & new_q if self._q else new_q
        return self.gen(self.path, q=combined)
