import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from finite_data import wafer
from fabroot.finite_visual import texture_tokens


class FiniteInputTests(unittest.TestCase):
    def test_pixel_tokens_are_observations(self):
        observed = wafer(np.zeros(8), np.random.default_rng(3))
        with patch.object(Image, "open", return_value=observed):
            tokens = texture_tokens("in-memory.png")
        self.assertEqual(tokens.shape, (4, 768))
        self.assertTrue(np.isfinite(tokens).all())
        self.assertTrue(np.all((tokens >= 0) & (tokens <= 1)))
        parameters = np.zeros(8)
        parameters[3] = 1
        changed = wafer(parameters, np.random.default_rng(3))
        self.assertFalse(np.array_equal(np.asarray(changed), np.asarray(wafer(np.zeros(8), np.random.default_rng(3)))))
