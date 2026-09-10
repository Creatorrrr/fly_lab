"""OS-owned worker leases: a crashed process releases its lock automatically."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

def acquire(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        if os.name == 'nt':
            # All handles lock the first byte, including an initially empty file.
            msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
        else:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
            raise ValueError('A worker already owns this campaign lease') from None
        raise
    return fd

def locked(path):
    try:fd=acquire(path)
    except ValueError:return True
    os.close(fd);return False

@contextmanager
def lease(path):
    fd=acquire(path)
    try:yield fd
    finally:os.close(fd)
