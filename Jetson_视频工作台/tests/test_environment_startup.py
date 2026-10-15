"""Resource failure must never launch an expensive TensorRT engine rebuild."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from runtime.backend import GPUResourceError
from check_environment import check


class EnvironmentTests(unittest.TestCase):
    def test_out_of_memory_preserves_engine_without_build_process(self):
        with patch('runtime.backend.TRTDetector',side_effect=GPUResourceError('fixture memory shortage')), \
             patch('check_environment.subprocess.run',return_value=SimpleNamespace(stdout='libx264')) as run:
            with self.assertRaises(GPUResourceError):check()
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.args[0],['ffmpeg','-hide_banner','-encoders'])


if __name__=='__main__':unittest.main()
