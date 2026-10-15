"""Ground-truth-backed, per-video COCO bbox AP; independent of tracking IDs."""
import contextlib
import hashlib
import json
import math
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def annotation_path(source, explicit=''):
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise ValueError('标注路径必须是本机绝对路径。')
        return path
    return Path(source).with_suffix('.gt.json')


def validate_ground_truth(data, source, mode, frames, size, names):
    info = data.get('info', {})
    if info.get('mode') != mode:
        raise ValueError('标注模式与当前模型不匹配，请选择 COCO/无人机的对应标注。')
    if info.get('video_sha256') != file_sha256(source):
        raise ValueError('标注绑定的视频 SHA256 不匹配，不能评估另一段或剪辑后的视频。')
    if info.get('frame_count') != frames:
        raise ValueError('标注帧数与实际处理帧数不一致。')
    images = data.get('images', [])
    image_ids = [image['id'] for image in images]
    image_set = set(image_ids)
    if len(image_ids) != frames or set(image_ids) != set(range(1, frames + 1)):
        raise ValueError('标注 images 必须覆盖全部帧，帧号从 1 开始且不重复。')
    if any([image['width'], image['height']] != list(size) for image in images):
        raise ValueError('标注尺寸与视频不一致，缩放视频必须同步缩放标注。')
    categories = data.get('categories', [])
    ids = [category['id'] for category in categories]
    labels = [category['name'] for category in categories]
    if not ids or len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise ValueError('标注类别为空或重复。')
    if any(label not in names for label in labels):
        raise ValueError('标注类别不能与模型标签对应，请先转换类别映射。')
    annotations = data.get('annotations', [])
    annotation_ids = set()
    for annotation in annotations:
        if annotation['id'] in annotation_ids:
            raise ValueError('标注 ID 重复。')
        annotation_ids.add(annotation['id'])
        if annotation['image_id'] not in image_set or annotation['category_id'] not in ids:
            raise ValueError('标注的帧号或类别无效。')
        box = annotation['bbox']
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box) or min(box[2:]) <= 0:
            raise ValueError('标注框必须是有效的 [x,y,width,height]。')
        if annotation.get('iscrowd', 0) not in (0, 1):
            raise ValueError('iscrowd 必须是 0 或 1。')
        annotation.setdefault('area', box[2] * box[3])
        annotation.setdefault('iscrowd', 0)
    if not any(not a['iscrowd'] for a in annotations):
        raise ValueError('没有可用于评估的有效标注目标。')
    return {names.index(category['name']): category['id'] for category in categories}


def evaluate_detections(data, predictions, cancel=None, use_native=True):
    """Use the reference pycocotools evaluator (101 recall points, IoU .50:.95)."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ground_truth = COCO()
    ground_truth.dataset = data
    ground_truth.createIndex()
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {'images': data['images'], 'categories': data['categories'], 'annotations': []}
        detections.createIndex()
    if use_native:
        from fast_coco import evaluator_class
        evaluator = evaluator_class()(ground_truth, detections, 'bbox')
    else:
        evaluator = COCOeval(ground_truth, detections, 'bbox')
    if cancel is not None:
        def interruptible(method):
            def call(*args, **kwargs):
                if cancel.is_set():
                    raise InterruptedError('评估已停止。')
                return method(*args, **kwargs)
            return call
        evaluator.computeIoU = interruptible(evaluator.computeIoU)
        evaluator.evaluateImg = interruptible(evaluator.evaluateImg)
    evaluator.params.imgIds = sorted(ground_truth.imgs)
    evaluator.params.catIds = sorted(ground_truth.cats)
    evaluator.evaluate()
    evaluator.accumulate()
    if cancel is not None and cancel.is_set():
        raise InterruptedError('评估已停止。')
    evaluator.summarize()
    precision = evaluator.eval['precision']  # IoU, recall, category, area, maxDets
    per_class = []
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        valid = values[values >= 0]
        per_class.append({'name': ground_truth.cats[category]['name'],
                          'map50_95': float(np.mean(valid)) if valid.size else None})
    return {'map50_95': float(evaluator.stats[0]), 'map50': float(evaluator.stats[1]),
            'map75': float(evaluator.stats[2]), 'recall100': float(evaluator.stats[8]),
            'per_class': per_class,
            'evaluation_backend': 'native matching + pycocotools COCOeval' if type(evaluator) is not COCOeval else 'pycocotools COCOeval'}


def evaluate_video(source, mode, frames, size, names, detections_path, output,
                   explicit='', confidence=.05, cancel=None):
    """Evaluation errors never turn a successfully encoded video into a fake success score."""
    output = Path(output)
    result = {'status': 'unavailable', 'message': '没有配套标注，无法计算 mAP；置信度不代表准确率。',
              'map50': None, 'map50_95': None}
    try:
        path = annotation_path(source, explicit)
        if not path.is_file():
            if explicit:
                raise ValueError('指定的标注 JSON 文件不存在。')
            result['annotation_path'] = str(path)
            return result
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError('标注文件超过 256 MiB，请拆分评估视频。')
        data = json.loads(path.read_text())
        category_map = validate_ground_truth(data, source, mode, frames, size, names)
        if cancel is not None and cancel.is_set():
            result.update(status='cancelled', message='评估已停止。')
            return result
        (output / 'ground_truth.json').write_text(json.dumps(data, ensure_ascii=False))
        predictions = []
        with Path(detections_path).open() as stream:
            for expected, line in enumerate(stream, start=1):
                record = json.loads(line)
                if record['frame'] != expected or expected > frames:
                    raise ValueError('检测日志帧号不连续。')
                for x1, y1, x2, y2, score, cls in record['detections']:
                    cls = int(cls)
                    if cls in category_map:
                        predictions.append({'image_id': expected, 'category_id': category_map[cls],
                                            'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score})
                if cancel is not None and cancel.is_set():
                    result.update(status='cancelled', message='评估已停止。')
                    return result
            if expected != frames:
                raise ValueError('检测日志没有覆盖全部视频帧。')
        # COCOeval emits its standard diagnostic table here, not into the UI.
        with (output / 'evaluation.log').open('w') as log, contextlib.redirect_stdout(log):
            scores = evaluate_detections(data, predictions, cancel=cancel)
        result = {**scores, 'status': 'completed', 'message': '已用人工标注评估原始检测框；不使用轨迹框。',
                  'frames': frames, 'categories': [c['name'] for c in data['categories']],
                  'annotation_path': str(path), 'annotation_sha256': file_sha256(path),
                  'valid_objects': sum(not a['iscrowd'] for a in data['annotations']),
                  'method': 'pycocotools COCO bbox AP, 101 recall points, maxDets=100 per image per category',
                  'confidence_min': confidence, 'scope': data['info'].get('description', '当前视频及标注类别'),
                  'note': '这是当前视频的检测 AP，不能代表整个数据集或 BoT-SORT 的跟踪准确率。'}
    except InterruptedError:
        result.update(status='cancelled', message='评估已停止。')
    except Exception as error:
        result.update(status='invalid', message='mAP 未计算：' + str(error))
    finally:
        (output / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result
