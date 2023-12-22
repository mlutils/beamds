class BeamSchema(object):

    def __init__(self, read_schema=None, write_schema=None, **kwargs):
        self._schema = {}
        self._schema.update(kwargs)
        self._read_schema = read_schema or {}
        self._write_schema = write_schema or {}

    def __getitem__(self, item):

        if item in self._schema.keys():
            return self._schema[item]
        elif item in self._read_schema.keys():
            return self._read_schema[item]
        elif item in self._write_schema.keys():
            return self._write_schema[item]
        else:
            raise KeyError(f'No schema for {item}')

    @property
    def read_schema(self):
        return {**self._schema, **self._read_schema}

    @property
    def write_schema(self):
        return {**self._schema, **self._write_schema}
