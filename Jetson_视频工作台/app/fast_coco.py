"""COCOeval interface with optional native matching; identical score definition.

Single-image adapter follows pycocotools COCOeval (BSD-2-Clause).
Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin. See runtime/native/README.md.
"""
import ctypes
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval


try:
    LIB = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / 'runtime/native/coco_match.so'))
    LIB.coco_match_abi.restype = ctypes.c_int
    if LIB.coco_match_abi() != 2:
        raise OSError('COCO matching ABI mismatch')
    LIB.coco_match.argtypes = [ctypes.c_size_t]*3 + [ctypes.c_void_p]*9
    LIB.coco_match.restype = None
except (OSError, AttributeError):
    LIB = None


class NativeCOCOeval(COCOeval):
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        p = self.params
        if p.useCats:
            gt, dt = self._gts[imgId,catId], self._dts[imgId,catId]
        else:
            gt = [g for c in p.catIds for g in self._gts[imgId,c]]
            dt = [d for c in p.catIds for d in self._dts[imgId,c]]
        if not gt and not dt:
            return None
        for g in gt:
            g['_ignore'] = int(bool(g['ignore']) or g['area'] < aRng[0] or g['area'] > aRng[1])
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')[:maxDet]
        dt = [dt[i] for i in dtind]
        ious = self.ious[imgId,catId]
        levels = np.ascontiguousarray(p.iouThrs, dtype=np.float64)
        T, G, D = len(levels), len(gt), len(dt)
        gtm, dtm = np.zeros((T,G)), np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt], dtype=np.uint8)
        dtIg = np.zeros((T,D), dtype=np.uint8)
        gtIds = np.array([g['id'] for g in gt], dtype=np.float64)
        dtIds = np.array([d['id'] for d in dt], dtype=np.float64)
        if len(ious):
            ious = np.ascontiguousarray(ious[:D, gtind], dtype=np.float64)
            crowd = np.array([g['iscrowd'] for g in gt], dtype=np.uint8)
            LIB.coco_match(T,D,G, *[a.ctypes.data for a in (ious,levels,crowd,gtIg,dtIds,gtIds,dtm,gtm,dtIg)])
        outside = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt], dtype=bool)[None,:]
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, outside))
        return dict(image_id=imgId,category_id=catId,aRng=aRng,maxDet=maxDet,
                    dtIds=[d['id'] for d in dt],gtIds=[g['id'] for g in gt],dtMatches=dtm,gtMatches=gtm,
                    dtScores=[d['score'] for d in dt],gtIgnore=gtIg,dtIgnore=dtIg)


def evaluator_class():
    return NativeCOCOeval if LIB is not None else COCOeval
