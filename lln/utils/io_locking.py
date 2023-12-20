import platform
import time
import pandas as pd

import platform

def lock_file(file, exclusive=True):
    if platform.system() == 'Windows':
        import msvcrt
        if exclusive:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBRLCK, 1)
    else:
        import fcntl
        if exclusive:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)

def unlock_file(file):
    if platform.system() == 'Windows':
        import msvcrt
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        
def read_pandas_df(file_path):
    while True:
        try:
            file = open(file_path, 'r')
            df = pd.read_csv(file_path)
            lock_file(file, exclusive=False)
            return file, df
        except IOError:
            print("File is unavailable for reading, waiting...")
            time.sleep(1)
