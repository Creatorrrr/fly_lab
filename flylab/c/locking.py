"""OS-owned worker leases: a crashed process releases its lock automatically."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

def acquire(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise ValueError('A worker already owns this campaign lease') from None
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
