"""
Event-driven profiler, designed around the POV of the cycle-time.  (I.e. of some outer loop)
With plots.
"""
import time
from threading import get_ident, Lock, Thread
import enum

from plot_data import plot_profile_data
from events import EventTypes


class WrongThreadException(Exception):
    pass


class LoopPerfTimer(object):
    """
    Class to time functions, collect other timing events, plot results.

    """

    def __init__(self):
        raise Exception("Call LoopPerfTimer's methods statically.  Do not instantiate.")

    _main_thread_id = get_ident()
    _enabled = True
    _loop_index = -1
    _events = []
    _disable_time = None
    _n_func_calls_started = 0
    _lock = Lock()
    _n_func_calls_finished = 0

    _burn_in = 0

    @staticmethod
    def _check_ident():
        """
        Make sure we are in main_thread, optionally setting it so if it's unset, else raise exception
        :param make_main: Define this thread as "main" if not already so.
        """
        ident = get_ident()
        if LoopPerfTimer._main_thread_id != ident:
            raise WrongThreadException("Only call from main thread!")

    @staticmethod
    def reset(enable=False, burn_in=0):
        """
        Clear all events, optionally start collecting events.
        """
        with LoopPerfTimer._lock:
            LoopPerfTimer._burn_in = burn_in
            _LOOP_INDEX = -1
            LoopPerfTimer._events = []
            LoopPerfTimer._enabled = enable

    @staticmethod
    def disable():
        """
        Stop collecting events.
        """
        with LoopPerfTimer._lock:
            LoopPerfTimer._disable_time = time.perf_counter()
            LoopPerfTimer._enabled = False

    @staticmethod
    def enable(burn_in=0):
        """
        Start/resume collecting events.
        """
        with LoopPerfTimer._lock:
            LoopPerfTimer._burn_in = burn_in
            LoopPerfTimer._enabled = True

    @staticmethod
    def _add_event(event_type, index, t, **kwargs):
        """
        :param event_type: EventTypes enum
        :param index:  loop index,( possibly different from current)
        :param kwargs:  Other event key:values
        :returns: current thread id, time
        """
        ident = get_ident()

        if LoopPerfTimer._loop_index < LoopPerfTimer._burn_in:
            return ident, t
        with LoopPerfTimer._lock:
            if LoopPerfTimer._enabled:
                LoopPerfTimer._events.append(dict(thread_id=ident,
                                                  time=t,
                                                  loop_index=index,
                                                  type=event_type,
                                                  **kwargs))
        return ident, t

    @staticmethod
    def mark_loop_start():
        """
        Call at the beginning of every loop, to align plots, etc.
        Always call from the same thread.  (this defines the "main" thread)
        """
        LoopPerfTimer._check_ident()
        LoopPerfTimer._loop_index += 1
        _, t_start = LoopPerfTimer._add_event(EventTypes.LOOP_START, LoopPerfTimer._loop_index,
                                              t=time.perf_counter())

    @staticmethod
    def time_function(func):
        """
        Decorator for functions to be timed.
        """

        def timed_func(*args, **kwargs):
            index = LoopPerfTimer._loop_index
            func_name = func.__qualname__
            LoopPerfTimer._n_func_calls_started += 1

            start = time.perf_counter()
            rv = func(*args, **kwargs)
            stop = time.perf_counter()

            LoopPerfTimer._n_func_calls_finished += 1
            LoopPerfTimer._add_event(EventTypes.FUNC_CALL,
                                     index,
                                     t=start,
                                     name=func_name,
                                     start_t=start,
                                     stop_t=stop)

            return rv

        return timed_func

    @staticmethod
    def add_marker(name):
        """
        Call to add a named marker, i.e. to plots
        """
        if not LoopPerfTimer._enabled:
            return
        LoopPerfTimer._add_event(EventTypes.MARKER,
                                 LoopPerfTimer._loop_index,
                                 name=name,
                                 t=time.perf_counter())

    @staticmethod
    def display_data():

        if LoopPerfTimer._enabled:
            raise Exception("Call mark_stop() before display_data().")
        plot_profile_data(events=LoopPerfTimer._events,
                          main_thread_id=LoopPerfTimer._main_thread_id,
                          burn_in=LoopPerfTimer._burn_in)


def perf_sleep(t):
    start = time.perf_counter()
    while time.perf_counter() - start < t:
        pass
    return time.perf_counter() - start
