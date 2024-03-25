import unittest

from pclib.nn.layers import FC

import torch
import torch.nn.functional as F


class TestLayers(unittest.TestCase):

    def test_fc_pred(self):
        layer = FC(
            in_features = 3,
            out_features = 2,
            has_bias = True,
            symmetric = True,
            actv_fn = F.relu,
            gamma = 0.1,
        )
        layer.weight.data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        layer.bias.data = torch.tensor([1, 2, 3], dtype=torch.float32)
        state = {'x': torch.tensor([[1, -2], [1, 3]], dtype=torch.float32), 'e': torch.zeros(2, 2, dtype=torch.float32)}
        pred = layer.predict(state)
        target = torch.tensor([[2, 4, 6], [14, 19, 24]], dtype=torch.float32)
        self.assertTrue(torch.allclose(pred, target))
    
    def test_fc_propagate(self):
        layer = FC(
            in_features = 3,
            out_features = 2,
            has_bias = True,
            symmetric = True,
            actv_fn = F.relu,
            gamma = 0.1,
        )
        layer.weight.data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        e = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        prop = layer.propagate(e)
        target = torch.tensor([[14, 32], [32, 77]], dtype=torch.float32)
        self.assertTrue(torch.allclose(prop, target))
    
    def test_fc_update(self):
        layer = FC(
            in_features = 3,
            out_features = 2,
            has_bias = True,
            symmetric = True,
            actv_fn = F.relu,
            gamma = 0.1,
            x_decay = 0.0,
        )
        layer.weight.data = torch.tensor([[-1, -2, 3], [4, 5, 6]], dtype=torch.float32)
        layer.bias.data = torch.tensor([1, 2, 3], dtype=torch.float32)
    
        state_l = {'x': torch.tensor([[1, 3, -1]], dtype=torch.float32), 'e': torch.zeros(1, 3, dtype=torch.float32)}
        state_lp1 = {'x': torch.tensor([[1, 3]], dtype=torch.float32), 'e': torch.zeros(1, 2, dtype=torch.float32)}

        pred = layer.predict(state_lp1)
        layer.update_e(state_l, pred)

        true_pred = torch.tensor([[12, 15, 24]], dtype=torch.float32)
        true_e_l = torch.tensor([[-11, -12, -25]], dtype=torch.float32)

        self.assertTrue(torch.allclose(pred, true_pred))
        self.assertTrue(torch.allclose(state_l['e'], true_e_l))

        layer.update_x(state_lp1, state_l['e'], torch.ones(1, dtype=torch.float32)*0.5)

        true_prop = torch.tensor([[-40, -294]], dtype=torch.float32)
        new_x_lp1 = torch.tensor([[-19, -144]], dtype=torch.float32)

        # print(state_lp1['x'])
        # print(new_x_lp1)
        self.assertTrue(torch.allclose(state_lp1['x'], new_x_lp1))


if __name__ == '__main__':
    unittest.main()