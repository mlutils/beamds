from beam.misc.fake import BeamFakeAlg


class DemoAlg:

    def __init__(self, sleep_time=1.):
        self.alg = BeamFakeAlg(sleep_time)

    def run(self, x):
        """
        Some fake explanation of the method run
        @param x: some input
        @return: a fake output
        """
        return self.alg.run(x)
