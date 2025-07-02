from .core import BeamIbis
from ..path import BeamURL


def _extract_backend_from_scheme(scheme):
    """
    Extract backend type from URL scheme.
    
    Handles various formats:
    - ibis-bigquery -> bigquery
    - ibis-postgresql -> postgresql  
    - mysql -> mysql
    - etc.
    """
    if not scheme:
        return scheme
    
    # Handle ibis-* prefixed schemes
    scheme = scheme.removeprefix('ibis-')
    
    # Handle direct schemes
    backend_mapping = {
        'postgres': 'postgresql',
        'mariadb': 'mysql',
    }
    
    return backend_mapping.get(scheme, scheme)


def _configure_backend_kwargs_and_path(backend, hostname, path, backend_kwargs):
    """
    Configure backend-specific parameters based on hostname, path, and backend type.
    
    This function handles common patterns where hostname or path contains important
    configuration information for different database backends.
    
    Supports both formats:
    - ibis-bigquery://project-name/dataset/table (hostname-based)
    - ibis-bigquery:///project-name/dataset/table (path-based)
    """
    # Make a copy to avoid modifying the original
    kwargs = backend_kwargs.copy()
    
    # BigQuery: project_id can come from hostname OR first path component
    if backend == 'bigquery':
        if 'project_id' not in kwargs:
            if hostname:
                # Format: ibis-bigquery://project-name/dataset/table
                path = f"/{hostname}/{path.lstrip('/')}"
        
        # Remove parameters that BigQuery doesn't accept
        kwargs.pop('host', None)
        kwargs.pop('hostname', None)
    
    # PostgreSQL/MySQL: hostname might contain database info
    elif backend in ['postgresql', 'mysql', 'mariadb'] and hostname:
        if 'host' not in kwargs:
            # If hostname looks like a connection string, use it
            if '.' in hostname or ':' in hostname:
                kwargs['host'] = hostname
    
    # SQLite: hostname or path might be the database file path
    elif backend == 'sqlite':
        if 'database' not in kwargs:
            if hostname and not hostname.startswith('http'):
                kwargs['database'] = hostname
            elif path and not path.startswith('http'):
                # For path-based SQLite: ibis-sqlite:///path/to/database.db
                kwargs['database'] = path.lstrip('/')
    
    # DuckDB: similar to SQLite
    elif backend == 'duckdb':
        if 'database' not in kwargs:
            if hostname and not hostname.startswith('http'):
                kwargs['database'] = hostname
            elif path and not path.startswith('http'):
                kwargs['database'] = path.lstrip('/')
    
    return kwargs, path


def beam_ibis(path, username=None, hostname=None, port=None, private_key=None, access_key=None, secret_key=None,
              password=None, scheme=None, backend=None, **kwargs):
    """
    Create a BeamIbis instance from a URL string.
    
    Args:
        path: URL string for the database connection
        username: Database username
        hostname: Database hostname
        port: Database port
        private_key: Private key for authentication
        access_key: Access key for authentication
        secret_key: Secret key for authentication  
        password: Database password
        scheme: URL scheme (e.g., 'bigquery', 'sqlite', 'postgresql')
        backend: Explicitly specify the backend type
        **kwargs: Additional connection parameters
    
    Returns:
        BeamIbis: Configured BeamIbis instance
    """
    url = BeamURL.from_string(path)

    # Extract connection info from URL
    if url.hostname is not None:
        hostname = url.hostname

    if url.port is not None:
        port = url.port

    if url.username is not None:
        username = url.username

    if url.password is not None:
        password = url.password

    # Parse query parameters
    query = url.query
    for k, v in query.items():
        kwargs[k.replace('-', '_')] = v

    # Handle authentication keys
    if access_key is None and 'access_key' in kwargs:
        access_key = kwargs.pop('access_key')
        
    if secret_key is None and 'secret_key' in kwargs:
        secret_key = kwargs.pop('secret_key')
        
    if private_key is None and 'private_key' in kwargs:
        private_key = kwargs.pop('private_key')

    # Determine path
    path = url.path
    if path == '':
        path = '/'

    fragment = url.fragment

    # Determine backend from scheme or explicit parameter
    if backend is None:
        if scheme is not None:
            backend = _extract_backend_from_scheme(scheme)
        elif url.scheme:
            backend = _extract_backend_from_scheme(url.scheme)

    # Add authentication parameters to backend_kwargs if provided
    backend_kwargs = kwargs.pop('backend_kwargs', {})
    if access_key is not None:
        backend_kwargs['access_key'] = access_key
    if secret_key is not None:
        backend_kwargs['secret_key'] = secret_key
    if private_key is not None:
        backend_kwargs['private_key'] = private_key
    
    # Configure backend-specific parameters generically
    backend_kwargs, path = _configure_backend_kwargs_and_path(backend, hostname, path, backend_kwargs)

    return BeamIbis(
        path,
        hostname=hostname, 
        backend=backend, 
        port=port, 
        username=username, 
        password=password,
        fragment=fragment, 
        backend_kwargs=backend_kwargs,
        **kwargs
    )

