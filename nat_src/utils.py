

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def ar_collate(x):
    return x if type(x) is dict else x[0]


class Tracker(object):

    def __init__(self, alpha, x=None):
        self.alpha = alpha
        self.x = 0 if x is None else x

    def update(self, xi):
        self.x = self.alpha * xi + (1 - self.alpha) * self.x
        return self.x
