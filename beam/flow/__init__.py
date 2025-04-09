if len([]):
    from .resource import airflow_client


__all__ = ['airflow_client']


def __getattr__(name):
    if name == 'airflow_client':
        from .resource import airflow_client
        return airflow_client
    raise AttributeError(f"module {__name__} has no attribute {name}")