"""TensorRT + CUDA runtime only. Inference does not import PyTorch or Ultralytics."""
import ctypes
import ctypes.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from runtime import frame_ops

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / 'vendor/botsort'
sys.path.insert(0, str(BOT))
cv2.setNumThreads(2)


class GPUResourceError(RuntimeError):
    """Resource shortage must not be treated as incompatible engine bytes."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class CUDA:
    """Small typed binding to the JetPack-provided CUDA runtime library."""
    def __init__(self):
        library = ctypes.util.find_library('cudart')
        if not library:
            candidates = list(Path('/usr/local').glob('cuda*/targets/aarch64-linux/lib/libcudart.so*'))
            if not candidates:
                candidates = list(Path('/usr/local').glob('cuda*/lib64/libcudart.so*'))
            library = str(candidates[0]) if candidates else 'libcudart.so.12'
        self.lib = ctypes.CDLL(library)
        pointer = ctypes.c_void_p
        signatures = {
            'cudaMalloc': [ctypes.POINTER(pointer), ctypes.c_size_t],
            'cudaFree': [pointer],
            'cudaHostAlloc': [ctypes.POINTER(pointer), ctypes.c_size_t, ctypes.c_uint],
            'cudaFreeHost': [pointer],
            'cudaMemcpy': [pointer, pointer, ctypes.c_size_t, ctypes.c_int],
            'cudaMemcpyAsync': [pointer, pointer, ctypes.c_size_t, ctypes.c_int, pointer],
            'cudaStreamCreate': [ctypes.POINTER(pointer)],
            'cudaStreamDestroy': [pointer],
            'cudaStreamSynchronize': [pointer],
            'cudaStreamBeginCapture': [pointer, ctypes.c_int],
            'cudaStreamEndCapture': [pointer, ctypes.POINTER(pointer)],
            'cudaGraphInstantiateWithFlags': [ctypes.POINTER(pointer), pointer, ctypes.c_ulonglong],
            'cudaGraphLaunch': [pointer, pointer],
            'cudaGraphExecDestroy': [pointer],
            'cudaGraphDestroy': [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code):
        if code:
            message = self.lib.cudaGetErrorString(code).decode()
            if code == 2:
                raise GPUResourceError(f'CUDA内存不足：{message}。请结束其他GPU任务后重试。')
            raise RuntimeError(f'CUDA {code}: {message}')

    def allocate(self, size, host=False):
        pointer = ctypes.c_void_p()
        if host:
            self.check(self.lib.cudaHostAlloc(ctypes.byref(pointer), size, 0))
        else:
            self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size))
        return pointer


class TRTDetector:
    def __init__(self, mode, size=None, engine=None, gpu_postprocess=False):
        import tensorrt as trt
        path = Path(engine) if engine else ROOT / f'models/{mode}_yolov12n_640_fp16.engine'
        metadata = json.loads(path.with_suffix('.meta.json').read_text())
        if metadata['mode'] != mode or (size is not None and metadata['size'] != size):
            raise ValueError('Engine mode/input size does not match its metadata')
        if sha256(path) != metadata['engine_sha256']:
            raise ValueError('Engine SHA256 does not match its metadata')
        if metadata.get('tensorrt') != trt.__version__:
            raise RuntimeError('TensorRT 版本不同，请运行 tools/build_engine.py 在目标 Jetson 重建引擎。')
        self.names = {int(key): value for key, value in metadata['names'].items()}
        self.metadata = metadata
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine 无法加载，请在本机重新构建。')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise GPUResourceError('无法创建TensorRT执行上下文，GPU可用内存可能不足；请释放内存后重新启动。')
        self.cuda = CUDA()
        self.stream = ctypes.c_void_p()
        self.graph = ctypes.c_void_p()
        self.graph_exec = ctypes.c_void_p()
        self.graph_status = 'disabled'
        self.allocations = []
        self.buffers = {}
        self.devices = {}
        self.closed = False
        self.reduce_lib = None
        try:
            self.cuda.check(self.cuda.lib.cudaStreamCreate(ctypes.byref(self.stream)))
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(self.engine.get_tensor_shape(name))
                if min(shape) <= 0:
                    raise ValueError('Only fixed shapes are supported')
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                size_bytes = int(np.prod(shape)) * dtype.itemsize
                host = self.cuda.allocate(size_bytes, host=True)
                self.allocations.append((host, True))
                device = self.cuda.allocate(size_bytes)
                self.allocations.append((device, False))
                view = np.ctypeslib.as_array((ctypes.c_ubyte * size_bytes).from_address(host.value))
                self.buffers[name] = view.view(dtype).reshape(shape)
                self.devices[name] = device
                self.context.set_tensor_address(name, device.value)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name
            input_shape = self.buffers[self.input_name].shape
            if input_shape[:2] != (1, 3) or len(input_shape) != 4:
                raise ValueError('Only batch-one RGB engines are supported')
            if list(input_shape) != metadata['input_shape']:
                raise ValueError('Engine input shape does not match metadata')
            self.input_size = input_shape[2:]
            self.size = max(self.input_size)
            self.dtype = self.buffers[self.input_name].dtype
            if gpu_postprocess and self.buffers[self.output_name].dtype==np.float32:
                try:
                    if len(self.names)<=1:raise ValueError('Single-class output needs no channel reduction')
                    lib=ctypes.CDLL(str(ROOT/'runtime/native/yolo_reduce.so'))
                    lib.yolo_reduce_abi.restype=ctypes.c_int
                    if lib.yolo_reduce_abi()!=1:raise RuntimeError('GPU reduction ABI mismatch')
                    lib.yolo_reduce.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
                    lib.yolo_reduce.restype=ctypes.c_int
                    shape=self.buffers[self.output_name].shape
                    if shape[:2]!=(1,4+len(self.names)):raise ValueError('Unsupported raw detector layout')
                    n=shape[2];count=6*n*4
                    host=self.cuda.allocate(count,host=True);self.allocations.append((host,True))
                    self.compact_device=self.cuda.allocate(count);self.allocations.append((self.compact_device,False))
                    view=np.ctypeslib.as_array((ctypes.c_ubyte*count).from_address(host.value))
                    self.compact_output=view.view(np.float32).reshape(1,6,n)
                    self.reduce_lib=lib
                except (OSError,AttributeError,RuntimeError,ValueError):
                    # Optional acceleration; the validated raw output path remains usable.
                    self.reduce_lib=None
        except Exception:
            self.close()
            raise

    def raw(self, tensor):
        self.submit(tensor)
        return self.synchronize()

    def submit(self, tensor):
        """Enqueue on the private stream; CPU can prepare the following frame."""
        if self.closed:
            raise RuntimeError('Detector is closed')
        inputs = self.buffers[self.input_name]
        np.copyto(inputs, tensor, casting='no')
        if self.graph_exec.value:
            self.cuda.check(self.cuda.lib.cudaGraphLaunch(self.graph_exec, self.stream))
        else:
            self._enqueue()
    def synchronize(self):
        self.cuda.check(self.cuda.lib.cudaStreamSynchronize(self.stream))
        return self.compact_output if self.reduce_lib else self.buffers[self.output_name]

    def _enqueue(self):
        inputs, outputs = self.buffers[self.input_name], self.buffers[self.output_name]
        self.cuda.check(self.cuda.lib.cudaMemcpyAsync(self.devices[self.input_name], inputs.ctypes.data,
                                                      inputs.nbytes, 1, self.stream))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError('TensorRT inference failed')
        if self.reduce_lib:
            self.cuda.check(self.reduce_lib.yolo_reduce(self.devices[self.output_name],self.compact_device,
                                                       outputs.shape[2],len(self.names),self.stream))
            self.cuda.check(self.cuda.lib.cudaMemcpyAsync(self.compact_output.ctypes.data,self.compact_device,
                                                         self.compact_output.nbytes,2,self.stream))
        else:
            self.cuda.check(self.cuda.lib.cudaMemcpyAsync(outputs.ctypes.data, self.devices[self.output_name],
                                                          outputs.nbytes, 2, self.stream))

    def enable_cuda_graph(self):
        """Fixed buffers/shapes permit replay without per-layer Python enqueue overhead.

        Call after ordinary warmup, before starting other pipeline threads. If an
        engine contains a non-capturable operation, keep the ordinary TRT path.
        """
        if self.graph_exec.value:
            return True
        capturing = False
        try:
            self.cuda.check(self.cuda.lib.cudaStreamSynchronize(self.stream))
            self.cuda.check(self.cuda.lib.cudaStreamBeginCapture(self.stream, 1))
            capturing = True
            self._enqueue()
            code = self.cuda.lib.cudaStreamEndCapture(self.stream, ctypes.byref(self.graph))
            capturing = False
            self.cuda.check(code)
            self.cuda.check(self.cuda.lib.cudaGraphInstantiateWithFlags(
                ctypes.byref(self.graph_exec), self.graph, 0))
            self.graph_status = 'enabled; fixed-shape H2D/TRT/D2H replay'
            return True
        except Exception as error:
            if capturing:
                self.cuda.lib.cudaStreamEndCapture(self.stream, ctypes.byref(self.graph))
            if self.graph_exec.value:
                self.cuda.lib.cudaGraphExecDestroy(self.graph_exec)
            if self.graph.value:
                self.cuda.lib.cudaGraphDestroy(self.graph)
            self.graph_exec, self.graph = ctypes.c_void_p(), ctypes.c_void_p()
            self.graph_status = 'fallback: ' + str(error)
            return False

    def close(self):
        if getattr(self, 'closed', True):
            return
        self.closed = True
        if self.stream.value:
            self.cuda.lib.cudaStreamSynchronize(self.stream)
        if self.graph_exec.value:
            self.cuda.lib.cudaGraphExecDestroy(self.graph_exec)
            self.graph_exec = ctypes.c_void_p()
        if self.graph.value:
            self.cuda.lib.cudaGraphDestroy(self.graph)
            self.graph = ctypes.c_void_p()
        for pointer, host in reversed(self.allocations):
            (self.cuda.lib.cudaFreeHost if host else self.cuda.lib.cudaFree)(pointer)
        self.allocations.clear()
        if self.stream.value:
            self.cuda.lib.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()
        self.context = self.engine = self.runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def preprocess(frame, size, dtype=np.float32):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR video frame')
    height, width = frame.shape[:2]
    input_h,input_w = size if isinstance(size,(tuple,list)) else (size,size)
    ratio = min(input_h / height, input_w / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    left = round((input_w - resized_width) / 2 - .1)
    top = round((input_h - resized_height) / 2 - .1)
    image = cv2.resize(frame, (resized_width, resized_height))
    if np.dtype(dtype) == np.float32:
        tensor = frame_ops.pack_rgb(image, size, top, left)
        if tensor is not None:
            return tensor, (ratio, left, top, height, width)
    image = cv2.copyMakeBorder(image, top, input_h - resized_height - top, left,
                              input_w - resized_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=dtype)
    tensor /= 255
    return tensor, (ratio, left, top, height, width)


def postprocess(raw, geometry, nc, conf=.05, iou=.7, max_det=300):
    if raw.shape[:2] != (1, 4 + nc):
        raise ValueError(f'Unexpected detector output shape {raw.shape}')
    channels = raw[0].astype(np.float32, copy=False)
    predictions = channels.T
    reduced = frame_ops.candidates(channels, nc, conf)
    if reduced is None:
        classes = np.argmax(predictions[:, 4:], axis=1)
        scores = predictions[np.arange(len(predictions)), 4 + classes]
        valid = np.isfinite(channels).all(axis=0) & (scores >= conf)
        valid &= (predictions[:, 2] > 0) & (predictions[:, 3] > 0)
        predictions, scores, classes = predictions[valid], scores[valid], classes[valid]
    else:
        indices, scores, classes = reduced
        predictions = predictions[indices]
    if not len(predictions):
        return np.empty((0, 6), dtype=np.float32)
    return _finish_postprocess(predictions,scores,classes,geometry,iou,max_det)


def _finish_postprocess(predictions,scores,classes,geometry,iou,max_det):
    boxes = np.column_stack((predictions[:, :2] - predictions[:, 2:4] / 2,
                             predictions[:, :2] + predictions[:, 2:4] / 2))
    selected = frame_ops.nms(boxes,scores,classes,iou,max_det)
    if selected is None:
        selected = []
        for category in np.unique(classes):
            positions = np.flatnonzero(classes == category)
            category_boxes = boxes[positions].copy()
            category_boxes[:, 2:] -= category_boxes[:, :2]
            kept = cv2.dnn.NMSBoxes(category_boxes.tolist(), scores[positions].tolist(), 0, iou)
            if len(kept):
                selected.extend(positions[np.asarray(kept).reshape(-1)].tolist())
    selected = np.asarray(selected, dtype=int)
    selected = selected[np.argsort(-scores[selected], kind='stable')[:max_det]]
    boxes, scores, classes = boxes[selected], scores[selected], classes[selected]
    ratio, left, top, height, width = geometry
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return np.column_stack((boxes, scores, classes))[valid].astype(np.float32)


def postprocess_pair(raw, geometry, nc, tracking_conf=.05, evaluation_conf=.001,
                     iou=.7, max_det=300):
    """Greedy NMS is descending-score: lower scores cannot suppress higher ones.

    Keep the same NMS, ordering and global cap for both consumers, evaluating it
    once at the lower threshold. No early top-k or candidate truncation is used.
    """
    detections = postprocess(raw, geometry, nc, min(tracking_conf, evaluation_conf), iou, max_det)
    return (detections[detections[:, 4] >= tracking_conf],
            detections[detections[:, 4] >= evaluation_conf])


def postprocess_compact_pair(raw,geometry,nc,tracking_conf=.05,evaluation_conf=.001,iou=.7,max_det=300):
    if raw.shape[:2]!=(1,6):raise ValueError('Invalid GPU-reduced output')
    p=raw[0].T
    valid=(p[:,4]>=min(tracking_conf,evaluation_conf))&(p[:,2]>0)&(p[:,3]>0)
    p=p[valid]
    if not len(p):d=np.empty((0,6),np.float32)
    else:d=_finish_postprocess(p,p[:,4],p[:,5].astype(np.int32),geometry,iou,max_det)
    return d[d[:,4]>=tracking_conf],d[d[:,4]>=evaluation_conf]


def tracker_args(with_reid=False, cmc='sparseOptFlow', **overrides):
    if with_reid:
        raise ValueError('该独立运行包使用 BoT-SORT 运动/位置关联，未启用未经验证的 ReID。')
    values = dict(track_high_thresh=.3, track_low_thresh=.05, new_track_thresh=.4,
                  track_buffer=30, match_thresh=.8, proximity_thresh=.5, appearance_thresh=.25,
                  with_reid=False, cmc_method=cmc, name='studio', ablation=False, mot20=False,
                  device='cuda:0', fast_reid_config='', fast_reid_weights='')
    values.update(overrides)
    return SimpleNamespace(**values)


def clipped_track_box(track, frame_shape):
    height, width = frame_shape[:2]
    box = np.asarray(track.tlbr, dtype=np.float64).copy()
    if not np.isfinite(box).all():
        raise ValueError('Non-finite track coordinates')
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    box[2:] -= box[:2]
    return box
