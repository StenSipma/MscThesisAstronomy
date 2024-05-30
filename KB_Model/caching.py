from pathlib import Path
from functools import wraps, partial
import pickle

# Allows caching of the decorated function (see test below on how to use)
def cache(func, name=None):
    if name is None:
        name = func.__name__
    filename = Path.home(
    ) / 'dataserver' / 'MscThesis' / 'data' / 'cache' / Path(name + '.pkl')
    filename = filename.resolve()

    @wraps(func)
    def inner(*args, force_run=False, **kwargs):
        if not force_run and filename.exists():
            print(f"Found: {filename.stem}, using cached result")
            with open(filename, 'rb') as f:
                result = pickle.load(f)
            return result
        else:
            if force_run:
                print(f"WARNING: running with force on, this may cause results to be overwritten!")
            else:
                print(f"{filename.stem} not found, running...")
            
            result = func(*args, **kwargs)
            
            print(f"Saving result to {filename}...")
            with open(filename, 'wb') as f:
                pickle.dump(result, f)
            print(f"Result saved")
            return result

    return inner


# Allows caching of the decorated function, with a specified filename for the cache
def named_cache(name):
    return partial(cache, name=name)
