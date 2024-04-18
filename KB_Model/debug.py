import numba

## Set the debug levels:
CRITICAL = 6
ERROR = 5
INFO = 4
WARN = 3
DEBUG = 2
DEBUGV = 1
DEBUGVV = 0

# Set the log/print levels here:
LOG_LEVEL = WARN

## Utility classes for debugging
@numba.jit(nopython=True, cache=True)
def log_critical(*args):
    if LOG_LEVEL <= CRITICAL:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_error(*args):
    if LOG_LEVEL <= ERROR:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_info(*args):
    if LOG_LEVEL <= INFO:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_warn(*args):
    if LOG_LEVEL <= WARN:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_debug(*args):
    if LOG_LEVEL <= DEBUG:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_debugv(*args):
    if LOG_LEVEL <= DEBUGV:
        print(*args)

@numba.jit(nopython=True, cache=True)
def log_debugvv(*args):
    if LOG_LEVEL <= DEBUGVV:
        print(*args)


