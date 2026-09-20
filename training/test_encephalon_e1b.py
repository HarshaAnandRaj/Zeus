"""Development-only E1-B credit, reproducibility, independent math and gate tests."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from core.encephalon_agent import Agent
from core.encephalon_world import World
from core.encephalon_resources import ResourceWorld
from training.encephalon_resource_reference import ResourceReference
from training.encephalon_e1 import deterministic,collect,tree_hash,read_checkpoint,write_checkpoint
from training.encephalon_e1b import Fit,evaluate
from training.run_encephalon_e1b import development_config
from training.encephalon_learning import loss
from training.audit_encephalon_e1 import Neural,portable,unpack
from training.audit_encephalon_e1b import replay,close,audit_decision
from training.encephalon_e1b_decision import decide


class E1B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):deterministic()

    def fit(self,arm="finite_32"):
        return Fit(arm,0,development_config(),True)

    def test_exact_twins_resume_and_portable_complete_state(self):
        for arm in ("finite_32","finite_128"):
            f=self.fit(arm)
            for _ in range(3):f.advance()
            midpoint=f.snapshot()
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/"state.pt";write_checkpoint(path,midpoint)
                g=Fit.restore(read_checkpoint(path))
                self.assertEqual(tree_hash(midpoint),tree_hash(unpack(portable(midpoint))))
            f.advance();g.advance();self.assertEqual(tree_hash(f.snapshot()),tree_hash(g.snapshot()))
            twin=self.fit(arm)
            for _ in range(4):twin.advance()
            self.assertEqual(tree_hash(f.snapshot()),tree_hash(twin.snapshot()))

    def test_resource_arms_start_with_same_model_and_lanes_do_not_change_role(self):
        for width in (32,128):
            a=self.fit(f"abundant_{width}");b=self.fit(f"finite_{width}")
            self.assertEqual(tree_hash(a.agent.state_dict()),tree_hash(b.agent.state_dict()))
            for fit in (a,b):
                for lane in range(fit.config["batch"]):
                    for _ in range(3):
                        w=fit.new_world(lane)
                        self.assertEqual(isinstance(w,World),lane<fit.config["original_lanes"])
                self.assertEqual(sum(fit.births),fit.created)

    def test_credit_reaches_every_module_and_changes_policy_at_both_widths(self):
        for arm in ("finite_32","finite_128"):
            f=self.fit(arm)
            public=torch.tensor([w.observation().values() for w in f.worlds],dtype=torch.float64)
            with torch.no_grad():before=f.agent(public,f.state)[0].clone()
            batch,_,bootstrap=collect(f.agent,f.worlds,f.state,f.sampler,8)
            total,_=loss(batch,bootstrap)
            first=torch.autograd.grad(total,batch["log_probability"],retain_graph=True)[0][0]
            modified=dict(batch,reward=batch["reward"].clone());modified["reward"][-1]+=.25
            altered,_=loss(modified,bootstrap)
            second=torch.autograd.grad(altered,batch["log_probability"])[0][0]
            self.assertFalse(torch.equal(first,second))
            f=self.fit(arm)
            with patch.object(ResourceReference,"act",side_effect=AssertionError("teacher action used")):
                metrics=f.advance()
            self.assertTrue(all(v>0 for v in metrics["module_gradient_norms"].values()))
            with torch.no_grad():after=f.agent(public,f.agent.initial(len(public)))[0]
            self.assertGreater(float((after-before).abs().max()),1e-7)

    def test_independent_carried_neural_state_through_real_updates(self):
        for arm in ("finite_32","finite_128"):
            f=self.fit(arm);independent=np.zeros((f.config["batch"],f.config["width"]))
            def checker(agent,obs,state,logits,value,fused,following):
                nonlocal independent
                s={k:v.detach().numpy().copy() for k,v in state.items()}
                independent[s["previous"]<0]=0.;s["h"]=independent.copy()
                n=Neural(agent.state_dict(),"recurrent")
                ll,vv,ff,hh=n.forward(obs.detach().numpy(),s)
                for a,b in ((ll,logits.detach().numpy()),(vv,value.detach().numpy()),(ff,fused.detach().numpy()),(hh,following["h"].detach().numpy())):close(a,b,f.config)
                actions=np.arange(len(obs))%6
                close(n.predict(ff,actions),agent.predict(fused,torch.from_numpy(actions)).detach().numpy(),f.config)
                alive=(obs[:,0]>0)&(obs[:,1]>0);independent[alive.numpy()]=hh[alive.numpy()]
            for _ in range(4):f.advance(checker)

    def test_independent_endpoints_and_corruptions_all_ecologies(self):
        c=development_config();c["endpoint_bodies"]=4
        for arm in ("finite_32","finite_128"):
            f=Fit(arm,0,c,True);f.advance()
            for ecology in c["ecologies"]:
                packet=evaluate(f.agent.state_dict(),arm,0,ecology,"balanced","trained",config=c,development=True)
                with patch.object(Agent,"forward",side_effect=AssertionError("primary neural called")),patch.object(World,"step",side_effect=AssertionError("primary physics called")),patch.object(ResourceWorld,"step",side_effect=AssertionError("primary resource physics called")):
                    replay(packet,f.agent.state_dict(),c,True)
                for field in ("actions","final","anchors","trace_sha256"):
                    broken=copy.deepcopy(packet)
                    if field=="actions":broken[field][0][0]=(broken[field][0][0]+1)%6
                    elif field=="final":broken[field][0]["energy"]+=1
                    elif field=="anchors":broken[field][0]["h"][0][0]+=.001
                    else:broken[field]="0"*64
                    with self.assertRaises(AssertionError):replay(broken,f.agent.state_dict(),c,True)
                if ecology=="finite":
                    broken=copy.deepcopy(packet);broken["stock_probes"][0]["empty"][0]+=.01
                    with self.assertRaises(AssertionError):replay(broken,f.agent.state_dict(),c,True)

    def synthetic(self):
        c=development_config();n=c["endpoint_bodies"];packets=[]
        for a in c["arms"]:
            for l in range(c["lineages"]):
                for e in c["ecologies"]:
                    for p in c["profiles"]:
                        for x in c["controls"]:
                            good=x=="trained"
                            packets.append(dict(arm=a,lineage=l,ecology=e,profile=p,control=x,
                                development=True,horizon=c["endpoint_horizon"],survived=[good]*n,
                                ticks=[c["endpoint_horizon"] if good else 40]*n,feeding=[int(good)]*n,
                                repairs=[int(good)]*n,food_energy=[[int(good)]*2 for _ in range(n)],action_counts=[[0]*6 for _ in range(n)]))
        return c,packets

    def test_canonical_complete_cells_independent_decision_and_no_hidden_bad_lineage(self):
        c,packets=self.synthetic();d=decide(packets,c,True)
        self.assertEqual(d,decide(list(reversed(packets)),c,True))
        audit_decision(list(reversed(packets)),d,c,True)
        self.assertEqual(d["e1b_verdict"],"PASS")
        self.assertTrue(all(v=="FAIL" for v in d["resource_benefit"].values()))
        for change in ("missing","duplicate","interval","decision","summary"):
            ps=copy.deepcopy(packets);bad=copy.deepcopy(d)
            if change=="missing":ps.pop()
            elif change=="duplicate":ps.append(ps[0])
            elif change=="interval":next(iter(bad["contrasts"].values()))["lower"]+=.01
            elif change=="decision":bad["e1b_verdict"]="FAIL"
            else:bad["cells"][0]["survived"]+=1
            with self.assertRaises(AssertionError):audit_decision(ps,bad,c,True)
        for p in packets:
            if p["control"]=="trained" and p["lineage"]==0 and p["ecology"]=="original" and p["profile"]=="energy":p["survived"][:2]=[False,False]
        d=decide(packets,c,True);audit_decision(packets,d,c,True)
        self.assertEqual(d["e1b_verdict"],"FAIL")


if __name__=="__main__":unittest.main()
