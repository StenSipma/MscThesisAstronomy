import pickle
import contextlib
import io

from pathlib import Path
from functools import wraps, partial


def cache_decorator(func, name=None, path=None):
    """
    Decorator which allows caching of the decorated function. Will write the 
    cache in the location: path / name, where the default name is the __name__
    of the function being decorated.
    
    To rerun the function pass 'force_run=True' to the function call.
    
    The returned result will contain a dictionary with:
        - result: the function result
        - stdout, stderr: str
        - args: list, of args passed to the function
        - kwargs: dict, of kwargs passed to the function
    """
    if name is None:
        name = func.__name__

    if path is None:
        raise ValueError("Specify a path to cache the results")

    filename = path / Path(name + '.pkl')
    filename = filename.resolve()

    @wraps(func)
    def inner(*args, force_run=False, **kwargs):
        if not force_run and filename.exists():
            print(f"Found: {filename.stem}, using cached result")
            with open(filename, 'rb') as f:
                stored = pickle.load(f)
            return stored
        else:
            if force_run:
                print(
                    f"WARNING: running with force on, this may cause results to be overwritten!"
                )
            else:
                print(f"{filename.stem} not found, running...")

            # Capture both the stderr and stdout in separate StringIO objects and save
            with contextlib.redirect_stdout(io.StringIO()) as stdout_str:
                with contextlib.redirect_stderr(io.StringIO()) as stderr_str:
                    func_result = func(*args, **kwargs)

            result = {
                'result': func_result,
                'stdout': stdout_str.getvalue(),
                'stderr': stderr_str.getvalue(),
                'args': args,
                'kwargs': kwargs,
            }

            print(f"Saving result to {filename}...")
            with open(filename, 'wb') as f:
                pickle.dump(result, f)
            print(f"Result saved")
            return result

    return inner


# Allows caching of the decorated function, with a specified filename for the cache
def named_cache(name, cache_decorator=cache_decorator):
    return partial(cache_decorator, name=name)


def init_cache(path):
    if isinstance(path, str):
        path = Path(path)
    return partial(cache_decorator, path=path)


class Cache:
    """
    Helper class to easier manage where cache is stored. Use like:
    
    ```python3
    # Initialize storage in current working directory (not recommended)
    cache = Cache(path='.')
    
    # Cache is stored at './func.pkl'
    @cache
    def func(a):
        # Some very expensive computation :o
        print(a)
        return a**2
        
    result = func(10)
    
    print(result['result'])  # will give: 100
    print(result['stdout'])  # will give: "10\n"
    print(result['stderr'])  # will give: ""
    print(result['args'])    # will give: [10]
    print(result['kwargs'])  # will give: {}
    ```
    
    Or alternatively, when you want to give a custom name for the storage:
    
    # Cache is stored at './sqrt-func.pkl'
    @cache.named("sqrt-func")
    def foo(b):
        # Another very expensive computation
        print(b)
        return b ** 0.5
    
    result_foo = foo(25)
    print(result['result'])  # will give: 25
    """
    def __init__(self, path):
        self.cacher = init_cache(path)

    def __call__(self, *args, **kwargs):
        return self.cacher(*args, **kwargs)

    def named(self, name):
        return partial(self.cacher, name=name)
