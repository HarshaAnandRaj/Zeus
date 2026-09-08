import unittest
import numpy as np
from tools import obs2r_investigations_20260908 as R


class Correction(unittest.TestCase):
    def test_three_rows_keep_all_episodes(self):
        rng=np.random.default_rng(11);episodes=[rng.normal(size=(n,32)) for n in (3,7,12)]
        h=np.array([e[:3] for e in episodes]);self.assertEqual(h.shape,(3,3,32))
        d=R.P.decompose(h);self.assertLess(d['identity_error'],1e-12)


if __name__=='__main__':unittest.main()
