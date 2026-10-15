"""Known AP outcomes and safeguards against evaluating the wrong video."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
import threading
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from evaluation import evaluate_detections, evaluate_video, file_sha256, validate_ground_truth


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.video = self.directory / 'sample.mp4'
        self.video.write_bytes(b'fixture identity, evaluator does not decode video')
        self.gt = {'info': {'mode': 'coco', 'video_sha256': file_sha256(self.video), 'frame_count': 2},
                   'images': [{'id': i, 'width': 100, 'height': 100} for i in [1, 2]],
                   'categories': [{'id': 0, 'name': 'person'}],
                   'annotations': [{'id': i, 'image_id': i, 'category_id': 0,
                                    'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0} for i in [1, 2]]}

    def tearDown(self):
        self.temporary.cleanup()

    def score(self, predictions):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_detections(copy.deepcopy(self.gt), predictions)

    def predictions(self):
        return [{'image_id': i, 'category_id': 0, 'bbox': [10, 10, 20, 20], 'score': .9} for i in [1, 2]]

    def test_perfect_boxes_are_one(self):
        result = self.score(self.predictions())
        self.assertAlmostEqual(result['map50'], 1)
        self.assertAlmostEqual(result['map50_95'], 1)

    def test_native_matches_reference_with_crowds_ties_and_empty_images(self):
        import numpy as np
        rng = np.random.default_rng(20261015)
        data = {'info':{},'images':[{'id':i,'width':640,'height':480} for i in range(1,9)],
                'categories':[{'id':i,'name':str(i)} for i in range(3)],'annotations':[]}
        predictions = []
        for frame in range(1,8):
            for n in range(25):
                x,y,w,h = rng.uniform([0,0,5,5],[580,420,100,100])
                annotation = dict(id=len(data['annotations'])+1,image_id=frame,category_id=n%2,
                                  bbox=[x,y,w,h],area=w*h,iscrowd=int(n%7==0))
                data['annotations'].append(annotation)
                for _ in range(2):
                    predictions.append(dict(image_id=frame,category_id=n%3,
                       bbox=[x+rng.uniform(-10,10),y,w,h],score=float(rng.choice([.01,.5,.9]))))
        with contextlib.redirect_stdout(io.StringIO()):
            native = evaluate_detections(copy.deepcopy(data),copy.deepcopy(predictions))
            reference = evaluate_detections(copy.deepcopy(data),copy.deepcopy(predictions),use_native=False)
        self.assertIn('native',native.pop('evaluation_backend'))
        reference.pop('evaluation_backend')
        self.assertEqual(native,reference)

    def test_evaluation_honours_stop(self):
        cancel = threading.Event()
        cancel.set()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InterruptedError):
            evaluate_detections(copy.deepcopy(self.gt), self.predictions(), cancel=cancel)

    def test_unlabelled_class_does_not_dilute_mean_ap(self):
        self.gt['categories'].append({'id': 1, 'name': 'bus'})
        result = self.score(self.predictions())
        self.assertAlmostEqual(result['map50_95'], 1)
        self.assertIsNone(result['per_class'][1]['map50_95'])

    def test_no_predictions_are_zero(self):
        result = self.score([])
        self.assertEqual(result['map50'], 0)
        self.assertEqual(result['map50_95'], 0)

    def test_native_library_unavailable_uses_reference(self):
        import fast_coco
        with mock.patch.object(fast_coco, 'LIB', None):
            result = self.score(self.predictions())
        self.assertAlmostEqual(result['map50_95'], 1)
        self.assertEqual(result['evaluation_backend'], 'pycocotools COCOeval')

    def test_numeric_annotation_ids_preserve_reference_matches(self):
        self.gt['annotations'][0]['id'] = .5
        self.gt['annotations'][1]['id'] = 2.5
        with contextlib.redirect_stdout(io.StringIO()):
            native = evaluate_detections(copy.deepcopy(self.gt), self.predictions())
            reference = evaluate_detections(copy.deepcopy(self.gt), self.predictions(), use_native=False)
        native.pop('evaluation_backend');reference.pop('evaluation_backend')
        self.assertEqual(native,reference)

    def test_high_score_false_positive_reduces_ap(self):
        predictions = [{'image_id': 1, 'category_id': 0, 'bbox': [70, 70, 10, 10], 'score': .99}]
        result = self.score(predictions + self.predictions())
        self.assertAlmostEqual(result['map50'], 2 / 3)

    def test_partial_overlap_changes_strict_ap(self):
        predictions = self.predictions()
        for p in predictions:
            p['bbox'] = [14, 10, 20, 20]  # IoU=2/3, passes .50/.55/.60/.65 only.
        result = self.score(predictions)
        self.assertAlmostEqual(result['map50'], 1)
        self.assertAlmostEqual(result['map50_95'], .4)

    def test_crowd_ignore_does_not_add_false_positive(self):
        self.gt['annotations'].append({'id': 3, 'image_id': 1, 'category_id': 0,
                                       'bbox': [60, 60, 30, 30], 'area': 900, 'iscrowd': 1})
        result = self.score([{'image_id': 1, 'category_id': 0, 'bbox': [65, 65, 10, 10], 'score': .99}] + self.predictions())
        self.assertAlmostEqual(result['map50'], 1)

    def test_wrong_video_and_dimensions_rejected(self):
        self.gt['info']['video_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_ground_truth(self.gt, self.video, 'coco', 2, [100, 100], ['person'])
        self.gt['info']['video_sha256'] = file_sha256(self.video)
        with self.assertRaisesRegex(ValueError, '尺寸'):
            validate_ground_truth(self.gt, self.video, 'coco', 2, [200, 100], ['person'])

    def test_missing_annotations_no_fabricated_score(self):
        result = evaluate_video(self.video, 'coco', 2, [100, 100], ['person'],
                                self.directory / 'detections.jsonl', self.directory)
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['map50'])
        self.assertEqual(json.loads((self.directory / 'evaluation.json').read_text()), result)

    def test_incomplete_predictions_rejected(self):
        self.video.with_suffix('.gt.json').write_text(json.dumps(self.gt))
        detections = self.directory / 'detections.jsonl'
        detections.write_text(json.dumps({'frame': 1, 'detections': []}) + '\n')
        result = evaluate_video(self.video, 'coco', 2, [100, 100], ['person'], detections, self.directory)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['map50_95'])


if __name__ == '__main__':
    unittest.main()
