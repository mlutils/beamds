class Groups(object):

    def __init__(self, groupby_pointer):
        self.groupby_pointer = groupby_pointer
        self.groups = {}

    def __getitem__(self, ind):
        if ind not in self.groups:
            self.groups[ind] = self.groupby_pointer().get_group(ind)

        return self.groups[ind]


class Iloc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._iloc(ind)


class Loc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._loc(ind)


class Key(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._key(ind)


def return_none(*args, **kwargs):
    return None
