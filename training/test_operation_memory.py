import unittest
import torch
from core.operation_memory import OperationMemory,events


class OperationMemoryTests(unittest.TestCase):
    def test_delayed_reward_reaches_writer_reader_and_answer(self):
        torch.set_num_threads(1);m=OperationMemory()
        e=events(256,torch.Generator().manual_seed(41))
        r=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2))
        reward=(r['answer']==e.target).float();adv=reward-(reward.sum()-reward)/255
        loss=-(adv*(r['writer_logp']+r['reader_logp']+r['answer_logp'])).mean()
        loss.backward()
        for p in m.parameters():
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)
        self.assertEqual(r['query_tick'],134)

    def test_delay_has_no_hidden_bypass(self):
        m=OperationMemory();e=events(32,torch.Generator().manual_seed(1))
        a=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2),delay=128)
        b=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2),delay=1024)
        for k in ('answer','slot','writer_logp','reader_logp'):
            self.assertTrue(torch.equal(a[k],b[k]))
        self.assertGreater(b['query_tick'],a['query_tick'])
        with self.assertRaises(ValueError):m(e.keys,e.values,e.priority,e.query,torch.Generator(),delay=64)

    def test_target_is_not_a_policy_input(self):
        m=OperationMemory();e=events(32,torch.Generator().manual_seed(1))
        a=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2))
        e.target=1-e.target
        b=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(a['answer'],b['answer']))

    def test_zero_content_removes_stored_value(self):
        m=OperationMemory();e=events(32,torch.Generator().manual_seed(1))
        a=m(e.keys,e.values,e.priority,e.query,torch.Generator().manual_seed(2),content_control='zero')
        b=m(e.keys,-e.values,e.priority,e.query,torch.Generator().manual_seed(2),content_control='zero')
        self.assertTrue(torch.equal(a['answer'],b['answer']))


if __name__=='__main__':unittest.main()
