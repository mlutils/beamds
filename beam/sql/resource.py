from .core import BeamIbis
from ..path import BeamURL


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
            if '_' in scheme:
                backend = scheme.split('_')[1]
            else:
                backend = scheme
        elif url.scheme:
            # Extract backend from URL scheme
            if url.scheme.startswith('bigquery'):
                backend = 'bigquery'
            elif url.scheme.startswith('sqlite'):
                backend = 'sqlite'
            elif url.scheme.startswith('postgresql') or url.scheme.startswith('postgres'):
                backend = 'postgresql'
            elif url.scheme.startswith('mysql'):
                backend = 'mysql'
            elif url.scheme.startswith('duckdb'):
                backend = 'duckdb'
            else:
                backend = url.scheme

    # Add authentication parameters to backend_kwargs if provided
    backend_kwargs = kwargs.pop('backend_kwargs', {})
    if access_key is not None:
        backend_kwargs['access_key'] = access_key
    if secret_key is not None:
        backend_kwargs['secret_key'] = secret_key
    if private_key is not None:
        backend_kwargs['private_key'] = private_key

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

