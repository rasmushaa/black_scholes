from time import time


def my_timing(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        values = func(*args, **kwargs)
        t1 = time()
        print(f'Function: {func.__name__}')
        print(f'With args: [{args}, {kwargs}]')
        print(f'Took {t1 - t0:.3f}s to run.\n')
        return values
    return wrapper