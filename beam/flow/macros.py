from datetime import datetime, timedelta

from pybtex.richtext import str_repr

from ..base import BeamBase



class JinjaItem:

    def __init__(self, repr_str):
        self.repr_str = repr_str

    def __getitem__(self, item):
        return JinjaItem(f"{self.repr_str}[{item}]")

    def __getattr__(self, item):
        return JinjaItem(f"{self.repr_str}.{item}")

    def __call__(self, *args, **kwargs):
        args_str = ", ".join([f"'{arg}'" if isinstance(arg, str) else str(arg) for arg in args])
        kwargs_str = ", ".join(
            [f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in kwargs.items()])
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))  # Remove empty elements
        return JinjaItem(f"{self.repr_str}({all_args})")

    @property
    def format(self):
        return f"{{{{ {self.repr_str} }}}}"


class JinjaTime:
    def __init__(self, repr_str, time_format):
        self.time_format = time_format
        self.repr_str = repr_str

    @property
    def format(self):
        return f"{{{{ ({self.repr_str}).strftime('{self.time_format}') }}}}"

    @staticmethod
    def get_delta(other):
        if type(other) == timedelta:
            str_repr = f"macros.timedelta(seconds={other.total_seconds()})"
        else:
            str_repr = f"macros.timedelta(seconds={other})"
        return str_repr

    def __add__(self, other):
        return JinjaItem(f"{self.repr_str} + {self.get_delta(other)}")

    def __sub__(self, other):
        return JinjaItem(f"{self.repr_str} - {self.get_delta(other)}")



class JinjaMacros(BeamBase):

    def __init__(self, *args, time_format=None, **kwargs):
        super().__init__(*args, time_format=time_format, **kwargs)
        self.time_format = self.hparams.time_format


    @property
    def execution_date(self):
        return JinjaTime("execution_date", self.time_format)

    @property
    def next_execution_date(self):
        return JinjaTime("next_execution_date", self.time_format)

    @property
    def prev_execution_date(self):
        return JinjaTime("prev_execution_date", self.time_format)

    def pull(self, task_id, key=None):
        if key is None:
            str_repr = f"ti.xcom_pull(task_ids='{task_id}')"
        else:
            str_repr = f"ti.xcom_pull(task_ids='{task_id}', key='{key}')"
        return JinjaItem(str_repr)

    def push(self, task_id, value, key=None):
        if key is None:
            str_repr = f"ti.xcom_push(key='{task_id}', value={value})"
        else:
            str_repr = f"ti.xcom_push(key='{key}', value={value})"
        return JinjaItem(str_repr)

    def get_var(self, var_name):
        return JinjaItem(f"var.value.{var_name}")

    def set_var(self, var_name, value):
        return JinjaItem(f"var.set('{var_name}', {value})")

    @property
    def dag(self):
        return self["dag"]

    @property
    def macros(self):
        return self["macros"]

    @property
    def ds(self):
        return self["ds"]

    def random(self, seed=None):
        return JinjaItem(f"macros.random({seed})" if seed is not None else "macros.random()")

    def uuid(self):
        return JinjaItem("macros.uuid()")

    @property
    def run_id (self):
        return JinjaItem("run_id")

    def __getitem__(self, item):
        return JinjaItem(item)

