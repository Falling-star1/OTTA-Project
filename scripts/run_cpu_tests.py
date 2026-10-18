"""Run the current delivery's existing CPU tests without installing packages."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libs', type=Path, help='Optional directory containing existing compatible Python libraries')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    test_cache = tempfile.TemporaryDirectory(prefix='otta-cpu-tests-')
    os.environ.setdefault('HF_HOME', test_cache.name)
    if args.libs:
        if not args.libs.resolve().is_dir():
            parser.error('--libs must point to an existing directory')
        sys.path.insert(0, str(args.libs.resolve()))
    import torch
    import transformers
    import peft
    print({'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}, flush=True)
    if transformers.__version__ != '4.46.1' or peft.__version__ != '0.12.0':
        parser.error('Use Transformers 4.46.1 and PEFT 0.12.0; server details are in docs/环境与运行.md')
    bundle = Path(__file__).resolve().parents[1] / 'offline_tta/tlm_tent_full_3gpu_det_bundle'
    sys.path.insert(0, str(bundle))
    suite = unittest.defaultTestLoader.discover(str(bundle), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

if __name__ == '__main__':
    main()
