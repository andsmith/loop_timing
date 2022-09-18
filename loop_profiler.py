"""
Event-driven profiler, designed around the POV of the cycle-time.  (I.e. of some outer loop)
With plots.
"""
import time
from threading import get_ident, Lock, Thread
import enum
import sys
from .plot_data import plot_profile_data
from .events import EventTypes
import pickle


class WrongThreadException(Exception):
    pass


class LoopPerfTimer(object):
    """
    Class to time functions, collect other timing events, plot results.
    NOTE:  don't instantiate, just call static methods, etc
    """

    def __init__(self):
        raise Exception("Call LoopPerfTimer's methods statically.  Do not instantiate.")

    _main_thread_id = None
    _enabled = True
    _loop_index = -1
    _events = []
    _lock = Lock()
    _burn_in = 0
    _display_after = 0
    _save_file=None

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
    def reset(enable=False, burn_in=0, display_after=0, save_results=None):
        """
        Clear all events, settings & reset.
        :param enable:  Start collecting data as soon as complete
        :param burn_in:  throw away this many loops first
        :param display_after:  Plot then exit after this loop count.
        :param save_results:  Save to file instead of plotting.
            plot:  "python loop_profiler.py profile_data.pkl"
        """
        if burn_in > 0 and display_after <= burn_in:
            raise Exception("Can't display loop %i to loop %i." % (burn_in, display_after))
        with LoopPerfTimer._lock:
            LoopPerfTimer._burn_in = burn_in
            _LOOP_INDEX = -1
            LoopPerfTimer._events = []
            LoopPerfTimer._enabled = enable
            LoopPerfTimer._display_after = display_after
            LoopPerfTimer._main_thread_id = None
            LoopPerfTimer._save_file = save_results

    @staticmethod
    def disable():
        """
        Stop collecting events.
        """
        with LoopPerfTimer._lock:
            LoopPerfTimer._disabled_time = time.perf_counter()
            LoopPerfTimer._enabled = False

    @staticmethod
    def enable():
        """
        Start/resume collecting events.
        """
        with LoopPerfTimer._lock:
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
        if LoopPerfTimer._main_thread_id is None:
            with LoopPerfTimer._lock:  # shouldn't be calling this from more than one thread anyway
                LoopPerfTimer._main_thread_id = get_ident()
        LoopPerfTimer._check_ident()
        LoopPerfTimer._loop_index += 1
        if LoopPerfTimer._loop_index >= LoopPerfTimer._display_after:
            LoopPerfTimer.display_data()
            sys.exit()
        if LoopPerfTimer._loop_index >= LoopPerfTimer._burn_in - 1:
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

            if not LoopPerfTimer._enabled or LoopPerfTimer._loop_index < LoopPerfTimer._burn_in:
                return func(*args, **kwargs)

            print(func.__name__)

            start = time.perf_counter()
            rv = func(*args, **kwargs)
            stop = time.perf_counter()

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
        if not LoopPerfTimer._enabled or LoopPerfTimer._loop_index < LoopPerfTimer._burn_in:
            return
        LoopPerfTimer._add_event(EventTypes.MARKER,
                                 LoopPerfTimer._loop_index,
                                 name=name,
                                 t=time.perf_counter())

    @staticmethod
    def save_data():
        data = dict(events=LoopPerfTimer._events,
                    main_thread_id=LoopPerfTimer._main_thread_id,
                    burn_in=LoopPerfTimer._burn_in)
        with open(LoopPerfTimer._save_file, 'wb') as outfile:
            pickle.dump(data, outfile)
        print("Saved loop profile data to file:  %s" % (LoopPerfTimer._save_file,))

    @staticmethod
    def load_data(filename):
        with open(filename, 'rb') as infile:
            data = pickle.load(infile)
        LoopPerfTimer._events = data['events']
        LoopPerfTimer._main_thread_id = data['main_thread_id']
        LoopPerfTimer._burn_in = data['burn_in']

        print("Loaded %i loop profile events from file:  %s" % (len(LoopPerfTimer._events), filename,))

    @staticmethod
    def display_data():
        if LoopPerfTimer._save_file is not None:
            print("Saving data instead of plotting.")
            LoopPerfTimer.save_data()
        else:
            plot_profile_data(events=LoopPerfTimer._events,
                              main_thread_id=LoopPerfTimer._main_thread_id,
                              burn_in=LoopPerfTimer._burn_in)
        sys.exit()

def perf_sleep(t):
    start = time.perf_counter()
    while time.perf_counter() - start < t:
        pass
    return time.perf_counter() - start


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To plot results, run:  python loop_profile.py loop_profile_data.pkl")
        sys.exit()
    LoopPerfTimer.load_data(filename=sys.argv[1])
    LoopPerfTimer.display_data()