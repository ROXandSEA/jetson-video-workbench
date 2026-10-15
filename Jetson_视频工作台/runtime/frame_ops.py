"""Optional native frame operations, with a portable NumPy fallback."""
import ctypes
from pathlib import Path
import numpy as np
import cv2

def _library():
    try:
        lib = ctypes.CDLL(str(Path(__file__).parent/'native/frame_ops.so'))
        lib.frame_ops_abi.restype = ctypes.c_int
        if lib.frame_ops_abi() != 3: return None
        ptr = ctypes.c_void_p
        lib.yolo_candidates.argtypes = [ptr,ctypes.c_int,ctypes.c_int,ctypes.c_float,ptr,ptr,ptr,ptr]
        lib.yolo_candidates.restype = ctypes.c_int
        lib.pack_rgb.argtypes = [ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
        lib.pack_rgb.restype = None
        lib.detection_json.argtypes=[ptr,ctypes.c_int,ctypes.c_ulonglong,ptr,ctypes.c_int]
        lib.detection_json.restype=ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None

LIB = _library()

def _nms_library():
    try:
        lib=ctypes.CDLL(str(Path(__file__).parent/'native/nms_bridge.so'))
        lib.nms_reference_version.restype=ctypes.c_char_p
        if lib.nms_reference_version().decode()!=cv2.__version__:return None
        lib.nms_abi.restype=ctypes.c_int
        if lib.nms_abi()!=2:return None
        lib.classwise_nms.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_int,ctypes.c_void_p]
        lib.classwise_nms.restype=ctypes.c_int
        return lib
    except (OSError,AttributeError):return None
NMS_LIB=_nms_library()

class DetectionJSON:
    def __init__(self):self.capacity=0;self.buffer=None
    def encode(self,frame,detections):
        if LIB is None:return None
        rows=np.ascontiguousarray(detections,dtype=np.float32)
        if rows.ndim!=2 or rows.shape[1]!=6:return None
        capacity=max(128,len(rows)*6*32+128)
        if capacity>self.capacity:
            self.capacity=capacity;self.buffer=ctypes.create_string_buffer(capacity)
        n=LIB.detection_json(rows.ctypes.data,len(rows),frame,self.buffer,self.capacity)
        return self.buffer.raw[:n] if n>=0 else None

def nms(boxes,scores,classes,threshold,max_keep):
    if NMS_LIB is None or max_keep<=0:return None
    boxes=np.ascontiguousarray(boxes,dtype=np.float32)
    if not np.isfinite(boxes).all():return None
    scores=np.ascontiguousarray(scores,dtype=np.float32)
    classes=np.ascontiguousarray(classes,dtype=np.int32)
    indices=np.empty(len(scores),np.int32)
    n=NMS_LIB.classwise_nms(boxes.ctypes.data,scores.ctypes.data,classes.ctypes.data,len(scores),
                            int(classes.max())+1,threshold,max_keep,indices.ctypes.data)
    return indices[:n] if n>=0 else None

def candidates(channels, nc, threshold):
    if LIB is None or channels.dtype != np.float32 or not channels.flags.c_contiguous:
        return None
    n = channels.shape[1]
    scores = np.empty(n,np.float32); classes = np.empty(n,np.int32)
    valid = np.empty(n,np.uint8); indices = np.empty(n,np.int32)
    count = LIB.yolo_candidates(channels.ctypes.data,n,nc,threshold,scores.ctypes.data,
                                classes.ctypes.data,valid.ctypes.data,indices.ctypes.data)
    selected = indices[:count]
    return selected, scores[selected], classes[selected]

def pack_rgb(image, size, top, left):
    if LIB is None or not image.flags.c_contiguous: return None
    h,w = size if isinstance(size,(tuple,list)) else (size,size)
    result = np.empty((1,3,h,w),np.float32)
    LIB.pack_rgb(image.ctypes.data,*image.shape[:2],h,w,top,left,result.ctypes.data)
    return result
