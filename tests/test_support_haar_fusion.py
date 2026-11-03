import unittest

import numpy as np
import torch

from pvseg.support_haar_fusion import fit_from_support, haar_features, predict


class SupportHaarFusionTests(unittest.TestCase):
    def test_haar_and_dynamic_fusion_train_from_one_support(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :32] = 60
        rgb[:, 32:] = 180
        rgb[::2, ::2, 0] = 240
        dino = torch.zeros(1, 3, 4, 4)
        dino[:, 0, :, :2] = 1
        dino[:, 1, :, 2:] = 1
        geo = torch.zeros(1, 3, 4, 4)
        geo[:, 0] = 1
        geo[:, 2, :, 2:] = 1
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., :32] = 1
        self.assertEqual(haar_features(rgb).shape, (1, 6, 4, 4))
        task = fit_from_support(rgb, dino, geo, mask, steps=5)
        probability, prediction = predict(task, rgb, dino, geo)
        self.assertEqual(probability.shape, (64, 64))
        self.assertEqual(prediction.shape, (64, 64))
        self.assertEqual(len(task['models']), 4)
        self.assertTrue(0 <= task['threshold'] <= 1)
        self.assertGreater(float(task['models'][0]['fusion']['alpha_head.weight'].abs().sum()), 0)
        self.assertGreater(float(task['models'][0]['correction']['head.weight'].abs().sum()), 0)
        self.assertGreater(task['models'][0]['updated_fusion_tensors'], 0)
        self.assertGreater(task['models'][0]['updated_haar_tensors'], 0)
        self.assertNotIn('source_checkpoint', task)


if __name__ == '__main__':
    unittest.main()
