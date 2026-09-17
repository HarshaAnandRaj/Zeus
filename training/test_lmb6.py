import copy,contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from training import run_lmb6 as R,audit_lmb6 as A,lmb6_judgment as J,lmb6_fit_guard as G
from training.audit_lifetime_calibration import close


class AnchoredCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        cls.config=R.K.CONFIG|dict(training_base=232014000,calibration_base=232064000,evaluation_base=232114000,
            regression_base=232124000,training_ecologies=4,calibration_ecologies=4,evaluation_ecologies=4,
            regression_ecologies=4,body_horizon=16,updates=2,chunk=4,batch_base=232314000,
            action_base=232414000,regression_action_base=232424000,bootstrap_seed=232514000,bootstrap_draws=64)
        cls.judgment_config=R.K.CONFIG|dict(bootstrap_draws=256)

    def judge(self,endpoints,preps):
        result=R.decide(endpoints,preps,self.judgment_config)
        close(result,J.decide(endpoints,preps,self.judgment_config))
        return result

    def test_teacher_provenance_physics_and_reachability(self):
        cfg=self.config;data=R.S.demonstrations('balanced',cfg,base=cfg['calibration_base'],n=4)
        self.assertGreater(A.teacher_audit(data,cfg['calibration_base'],4,cfg),0)
        self.assertEqual(R.calibration_decide(data,cfg),A.calibration_decide(data,cfg))
        self.assertEqual(R.calibration_decide(data,cfg)['verdict'],'PASS')
        changed=copy.deepcopy(data);changed[0]['records'][2]['label']=0
        with self.assertRaises(AssertionError):A.teacher_audit(changed,cfg['calibration_base'],4,cfg)
        changed=copy.deepcopy(data);changed[0]['preparation']['energy']=.85
        with self.assertRaises(AssertionError):A.teacher_audit(changed,cfg['calibration_base'],4,cfg)

    def test_campaign_replay_nonzero_corrections_and_donors(self):
        cfg=self.config;preps=R.preparations(cfg['evaluation_base'],4,cfg)
        for variant in (*cfg['arms'],'warm'):
            body,_=R.S.initial(0) if variant=='warm' else R.B.initial(0,variant)
            if variant in ('current','recurrent'):
                with torch.no_grad():
                    body.anchor.weight.fill_(.023);body.anchor.bias.copy_(torch.linspace(-.06,.08,32))
            before=R.M.L.P.L3.tree_hash(body.state_dict())
            for prep in preps:
                for mode in ('inherited','empty'):
                    records=[]
                    actual=R.operate(body,0,variant,prep,mode,cfg,records.append)
                    # Disable production forward methods during the independent replay.
                    with patch.object(type(body),'logits',side_effect=AssertionError('production replay forbidden')):
                        replay,stats=A.body_audit(body.state_dict(),prep,records,0,variant,mode,body.parent_hash,cfg)
                    self.assertEqual(actual,replay);self.assertLess(stats['max_local_probability_error'],2e-5)
                    self.assertLess(stats['max_local_state_error'],1e-4)
            item=R.readout(body,0,preps,cfg)
            A.query_audit(body.state_dict(),item,preps,0,variant,body.parent_hash,cfg)
            changed=copy.deepcopy(item);changed['rows'][2]['used'][0][0]+=.01
            with self.assertRaises(AssertionError):A.query_audit(body.state_dict(),changed,preps,0,variant,body.parent_hash,cfg)
            with self.assertRaises(AssertionError):A.replayer(body.state_dict(),variant,'wrong-parent')
            if variant in ('current','recurrent'):
                other='recurrent' if variant=='current' else 'current'
                with self.assertRaises(AssertionError):A.replayer(body.state_dict(),other,body.parent_hash)
            self.assertEqual(before,R.M.L.P.L3.tree_hash(body.state_dict()))

    def abstract(self):
        cfg=self.judgment_config;preps=[]
        for e in cfg['energies']:
            for i in range(128):
                side=(i//2)%2;q=int(i%2==side);preps.append(dict(index=i,energy=e,side=side,quality=q,target=1+i%2))
        endpoints={}
        for variant in (*cfg['arms'],'warm'):
            bodies=[];queries=[]
            for trial in range(4):
                for p in preps:
                    modes=('inherited','empty') if variant!='warm' and p['energy']==.85 else ('inherited',)
                    for mode in modes:
                        bodies.append(dict(trial=trial,variant=variant,mode=mode,**p,survived=True,feeding=28,repairs=2,quality_correct=True,
                            initial_z=[1.]*8 if mode=='inherited' else [0.]*8,written_z=[1.]*8,storage_distance=0.,fast_reset=True))
                if variant=='warm':continue
                rows=[]
                for control in ('full','reset','opposite'):
                    source=preps if control!='opposite' else [preps[i^1] for i in range(len(preps))]
                    rows.append(dict(control=control,action=[p['target'] for p in source] if control!='reset' else [1]*512,
                        recall_probability=[.99 if p['quality'] else .01 for p in source]))
                queries.append(dict(trial=trial,storage_distance=0.,written=[[1.]*8]*512,inherited=[[1.]*8]*512,rows=rows))
            endpoints[variant]=dict(bodies=bodies,readout=queries)
        return endpoints,preps

    def test_cell_boundaries_selection_and_no_pooled_rescue(self):
        endpoints,preps=self.abstract();result=self.judge(endpoints,preps)
        self.assertEqual(result['selected_qualified_arm'],'current')
        self.assertEqual(result['architecture_attribution'],'FAIL')
        self.assertEqual(len(result['contrasts']),8);self.assertEqual(len(result['warm_contrasts']),12)
        self.assertEqual(sum(r['required'] for r in result['contrasts']),4)
        for arm,next_arm in (('current','recurrent'),('recurrent','legacy'),('legacy',None)):
            cell=[r for r in endpoints[arm]['bodies'] if (r['trial'],r['energy'],r['mode'],r['side'],r['quality'])==(3,.12,'inherited',1,0)]
            self.assertEqual(len(cell),32)
            for row in cell[:3]:row['survived']=False
            self.assertEqual(self.judge(endpoints,preps)['selected_qualified_arm'],arm)
            cell[3]['survived']=False
            self.assertEqual(self.judge(endpoints,preps)['selected_qualified_arm'],next_arm)
        self.assertEqual(self.judge(endpoints,preps)['verdict'],'FAIL')
        endpoints['warm']['bodies'][-1]=copy.deepcopy(endpoints['warm']['bodies'][0])
        for judge in (R.decide,J.decide):
            with self.assertRaises(AssertionError):judge(endpoints,preps,self.judgment_config)

    def test_architecture_claim_requires_both_comparators_both_scales_and_qualification(self):
        endpoints,preps=self.abstract()
        for control in ('recurrent','legacy'):
            for row in endpoints[control]['bodies']:
                if row['energy'] in (.12,.20):row['survived']=False
        self.assertEqual(self.judge(endpoints,preps)['architecture_attribution'],'PASS')
        for row in endpoints['recurrent']['bodies']:
            if row['energy']==.20:row['survived']=True
        self.assertEqual(self.judge(endpoints,preps)['architecture_attribution'],'FAIL')
        for row in endpoints['recurrent']['bodies']:
            if row['energy']==.20:row['survived']=False
        # Large scarce gains do not excuse a failed high-energy functional cell.
        cell=[r for r in endpoints['current']['bodies'] if (r['trial'],r['energy'],r['mode'],r['side'],r['quality'])==(0,.85,'empty',0,0)]
        for row in cell[:4]:row['survived']=False
        self.assertEqual(self.judge(endpoints,preps)['architecture_attribution'],'FAIL')

    def test_role_and_stage_guards_before_any_endpoint(self):
        R.K.validate();R.K.validate(self.config)
        with self.assertRaises(AssertionError):R.K.validate(R.K.CONFIG|dict(regression_base=R.K.CONFIG['evaluation_base']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unlaunched'
            with patch.object(R,'OUT',path),patch.object(R,'prerequisites',side_effect=AssertionError('missing source')):
                with self.assertRaises(AssertionError):R.prepare()
            self.assertFalse(path.exists())
        for receipt in (dict(status='FAIL',verdict='PASS'),dict(status='PASS',verdict='VOID')):
            with patch.object(R.M.L.P.L3,'read',return_value=receipt):
                with self.assertRaises(AssertionError):R.calibration_qualified()
        with patch.object(R,'verify',return_value={}),patch.object(R,'calibration_qualified'),patch.object(R,'complete_fits',side_effect=AssertionError('missing fit')),patch.object(R,'preparations') as unseen:
            with self.assertRaises(AssertionError):R.evaluate()
            unseen.assert_not_called()

    def test_actual_fit_guard_and_resealed_corruption(self):
        cfg=self.config|dict(trials=1);data=R.S.demonstrations('balanced',cfg)
        raw,heads=R.S.initial(0)
        bound=dict(source=dict(heads={'0':G.M.L.P.L3.tree_hash(heads.state_dict())}),
            parent_identities={'0':raw.parent_hash},initial_hashes={})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for twin in ('a','b'):R.C.save(path/f'training_{twin}.json',data)
            for arm in cfg['arms']:
                body,_=R.B.initial(0,arm);bound['initial_hashes']['0_'+arm]=G.M.L.P.L3.tree_hash(body.state_dict())
                for twin in ('a','b'):R.B.fit(0,arm,twin,data,cfg,path/f'0_{arm}_{twin}')
            manifest=dict(prerequisites=bound)
            receipt=G.verify_fits(path,manifest,cfg)
            self.assertEqual(len(receipt['fit_pairs']),3);self.assertEqual(receipt['training_batches_verified'],6)
            original,_=G.M.L.P.L3.checked_checkpoint(path/'0_current_a')
            def seal(payload):
                for twin in ('a','b'):
                    folder=path/f'0_current_{twin}';torch.save(payload,folder/'checkpoint.pt')
                    complete=dict(logical_hash=G.M.L.P.L3.tree_hash(payload),checkpoint_sha=G.M.L.P.L3.sha(folder/'checkpoint.pt'))
                    (folder/'completion.json').write_text(json.dumps(complete))
            corruptions=[]
            changed=copy.deepcopy(original);changed['model']['_extra_state']['mode']='recurrent';corruptions.append(changed)
            changed=copy.deepcopy(original);next(iter(changed['optimizer']['state'].values()))['step'].fill_(1);corruptions.append(changed)
            changed=copy.deepcopy(original);next(iter(changed['optimizer']['state'].values()))['exp_avg'].fill_(float('inf'));corruptions.append(changed)
            changed=copy.deepcopy(original);changed['logs'][0]['live_training_steps']+=1;corruptions.append(changed)
            changed=copy.deepcopy(original);changed['model']['anchor.weight'].zero_();changed['model']['anchor.bias'].zero_();corruptions.append(changed)
            for changed in corruptions:
                seal(changed)
                with self.assertRaises(AssertionError):G.verify_fits(path,manifest,cfg)
            seal(original)
            completion=path/'0_legacy_b/completion.json';saved=completion.read_bytes();completion.unlink()
            with self.assertRaises((FileNotFoundError,AssertionError)):G.verify_fits(path,manifest,cfg)
            completion.write_bytes(saved)
            self.assertEqual(G.verify_fits(path,manifest,cfg),receipt)

    def test_complete_development_pipeline_all_parents_and_arms(self):
        # The full source guard is checked separately on real historical evidence.
        # Here only that expensive receipt and git freeze are substituted; actual
        # generation, 24 fits, files, world/model replay and both judges execute.
        cfg=self.config;initials={};parents={};heads={}
        for trial in range(4):
            raw,hd=R.S.initial(trial);parents[str(trial)]=raw.parent_hash
            heads[str(trial)]=G.M.L.P.L3.tree_hash(hd.state_dict())
            for arm in cfg['arms']:
                body,_=R.B.initial(trial,arm);initials[f'{trial}_{arm}']=G.M.L.P.L3.tree_hash(body.state_dict())
        manifest=dict(commit='development-fixture',prerequisites=dict(source=dict(heads=heads),
            initial_hashes=initials,parent_identities=parents))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            reports=root/'zeus_sandbox/universe/reports';reports.mkdir(parents=True)
            R.C.save(out/'manifest.json',manifest)
            with patch.object(R,'OUT',out),patch.object(R,'REPORT',reports/'lmb6_20260914.json'),patch.object(A,'ROOT',root),patch.object(R,'SOURCES',()),patch.object(R.K,'CONFIG',cfg),patch.object(R,'verify',return_value=manifest),contextlib.redirect_stdout(io.StringIO()):
                R.calibrate();A.audit_calibration();R.train();R.evaluate();R.finalize();A.main()
                receipt=R.M.L.P.L3.read(reports/'lmb6_audit_20260914.json')
                self.assertEqual(receipt['status'],'PASS')
                self.assertEqual(receipt['endpoint_bodies'],304)
                self.assertEqual(receipt['readout_queries'],576)
                self.assertEqual(receipt['training_batches_verified'],24)
                self.assertEqual(len(receipt['fit_pairs']),12)
                self.assertEqual(receipt['decisions'],receipt['body_decisions']+receipt['readout_queries'])
                self.assertEqual(receipt['report_sha'],R.M.L.P.L3.sha(R.REPORT))
                self.assertFalse(receipt['pillar_promotion'])
                with self.assertRaises(FileExistsError):R.calibrate()


if __name__=='__main__':unittest.main()
