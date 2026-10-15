"""Profiles cannot leak modes or use another confidence for AP."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
from jobs import JobManager

class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        self.source=self.directory/'source.mp4';self.source.write_bytes(b'fixture')
        self.requests=[]
        def runner(request,publish,cancel):self.requests.append(request);return {}
        self.manager=JobManager(json.loads((ROOT/'app/config.json').read_text()),ROOT/'app',runner)
    def tearDown(self):self.temp.cleanup()
    def test_default_coco_preserves_low_track_threshold_and_separate_ap(self):
        self.manager.start(str(self.source),str(self.directory),'coco');self.manager.thread.join()
        req=self.requests[0];self.assertEqual(req['profile'],'quality');self.assertIn('1280_fp16',req['config']['engine'])
        self.assertEqual(req['config']['track_low_thresh'],.05);self.assertEqual(req['config']['evaluation_conf'],.001)
    def test_int8_profile_uses_int8_engine(self):
        self.manager.start(str(self.source),str(self.directory),'coco',profile='int8');self.manager.thread.join()
        self.assertIn('640_int8',self.requests[0]['config']['engine'])
    def test_int8_cannot_be_selected_for_uav(self):
        with self.assertRaises(ValueError):self.manager.start(str(self.source),str(self.directory),'uav',profile='int8')
    def test_package_model_paths_stay_inside_package(self):
        for mode,items in self.manager.config['profiles'].items():
            for cfg in items.values():self.assertTrue(self.manager.resolve(cfg['engine']).is_relative_to(ROOT))

if __name__=='__main__':unittest.main()
