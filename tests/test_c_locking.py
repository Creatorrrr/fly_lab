import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from flylab.c.locking import lease, locked


ROOT = Path(__file__).resolve().parents[1]
CHILD_OPTIONS = dict(cwd=ROOT, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))


class LockingTests(unittest.TestCase):
    def test_other_process_is_excluded_until_lease_closes(self):
        code = (
            'import sys\n'
            'from flylab.c.locking import lease\n'
            'try:\n'
            '    with lease(sys.argv[1]): pass\n'
            'except ValueError:\n'
            '    sys.exit(3)\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.lock'
            with lease(path):
                self.assertTrue(locked(path))
                result = subprocess.run([sys.executable, '-c', code, str(path)],
                                        capture_output=True, timeout=15, **CHILD_OPTIONS)
                self.assertEqual(result.returncode, 3, result.stderr)
            self.assertFalse(locked(path))
            result = subprocess.run([sys.executable, '-c', code, str(path)],
                                    capture_output=True, timeout=15, **CHILD_OPTIONS)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_process_exit_releases_lease_without_cleanup(self):
        code = (
            'import os, sys\n'
            'from flylab.c.locking import acquire\n'
            'fd = acquire(sys.argv[1])\n'
            'print("ready", flush=True)\n'
            'sys.stdin.read(1)\n'
            'os._exit(0)\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.lock'
            process = subprocess.Popen([sys.executable, '-c', code, str(path)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, **CHILD_OPTIONS)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                self.assertTrue(locked(path))
                process.communicate('x', timeout=15)
                self.assertEqual(process.returncode, 0)
                self.assertFalse(locked(path))
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=15)


if __name__ == '__main__':
    unittest.main()
