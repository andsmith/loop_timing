"""
Event-driven profiler, designed around the POV of the cycle-time.  (I.e. of some outer loop)
With plots.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import re

from threading import get_ident, Lock, Thread

import enum


class WrongThreadException(Exception):
    pass


class EventTypes(enum.Enum):
    LOOP_START = 0
    MARKER = 1
    FUNC_CALL = 2


class LoopPerfTimer(object):
    """
    Class to time functions, collect other timing events, plot results.
    """

    def __init__(self):
        """
        """
        self._main_thread_id = get_ident()
        self._enabled = False
        self._loop_index = -1
        self._events = []

        self._lock = Lock()

    def _check_ident(self):
        """
        Make sure we are in main_thread, optionally setting it so if it's unset, else raise exception
        :param make_main: Define this thread as "main" if not already so.
        """
        ident = get_ident()
        if self._main_thread_id != ident:
            raise WrongThreadException("Only call from main thread!")

    def reset(self, enable=False):
        """
        Clear all events, optionally start collecting events.
        """
        with self._lock:
            _LOOP_INDEX = -1
            self._events = []
            self._enabled = enable

    def disable(self):
        """
        Stop collecting events.
        """
        with self._lock:
            self._enabled = False

    def enable(self):
        """
        Start/resume collecting events.
        """
        print("Enabled", get_ident())
        with self._lock:
            self._enabled = True
        print("ed")

    def _add_event(self, event_type, index, **kwargs):
        """
        :param event_type: EventTypes enum
        :param index:  loop index,( possibly different from current)
        :param kwargs:  Other event key:values
        :returns: current thread id, time
        """
        ident = get_ident()
        t = time.perf_counter()

        if self._enabled:
            self._events.append(dict(thread_id=ident,
                                     time=t,
                                     loop_index=index,
                                     type=event_type,
                                     **kwargs))
        return ident, t

    def mark_loop_start(self):
        """
        Call at the beginning of every loop, to align plots, etc.
        Always call from the same thread.  (this defines the "main" thread)
        """
        self._check_ident()

        with self._lock:
            self._loop_index += 1
            _, t_start = self._add_event(EventTypes.LOOP_START, self._loop_index)

    def time_function(self, func):
        """
        Decorator for functions to be timed.
        """
        print("CALL", self._enabled, get_ident())

        def timed_func(*args, **kwargs):
            index = self._loop_index
            func_name = func.__name__

            start = time.perf_counter()
            rv = func(*args, **kwargs)
            stop = time.perf_counter()

            self._add_event(EventTypes.FUNC_CALL,
                            index,
                            name=func_name,
                            start=start,
                            stop=stop)
            return rv

        return timed_func

    def add_marker(self, name):
        """
        Call to add a named marker, i.e. to plots
        """
        if not self._enabled:
            print("FAIL")
            return
        self._add_event(EventTypes.MARKER,
                        self._loop_index,
                        name=name)
    """
    @staticmethod
    def display_data(print_avgs=True, plot=True):
        import ipdb;
        ipdb.set_trace()

        stop_event = LoopPerfTimer._events[-1]

        if LoopPerfTimer._ENABLED or stop_event['type'] != 'end':
            raise Exception("Call mark_stop() before display_data().")

        user_events = [event for event in LoopPerfTimer._EVENTS if event['type'] in ['function', 'marker']]
        loop_starts = [event for event in LoopPerfTimer._EVENTS if event['type'] == 'loop start']
        loop_indices = [e['loop_index'] for e in loop_starts]
        loop_reverse_index = {li: i for i, li in enumerate(loop_indices)}
        smallest = np.min(loop_indices)
        loop_reverse_index[smallest - 1] = loop_reverse_index[
            smallest]  # just in case something got added before the first loop started
        loop_start_times = [e['time'] for e in loop_starts]
        loop_end_times = loop_start_times[1:] + [stop_event['time']]
        loop_durations = np.array(loop_end_times) - np.array(loop_start_times)

        # re-sort, since order is originally by completion time
        def _event_sort_key(e):
            if e['type'] == 'marker':
                return e['time']
            elif e['type'] == 'function':
                return e['start']
            else:
                raise Exception("Unknown event type:  %s" % (e['type'],))

        user_events = sorted(user_events, key=_event_sort_key)

        # determine which functions/markers are in which threads
        thread_ids = list(set([event['thread_id'] for event in user_events]))
        # put main thread first
        thread_ids = [LoopPerfTimer._MAIN_THREAD_ID] + [t_id for t_id in thread_ids
                                                        if t_id != LoopPerfTimer._MAIN_THREAD_ID]
        n_threads = len(thread_ids)
        n_loops = len(loop_starts)
        thread_spacing = 1.0 / (.5 + n_threads)
        thickness = 250 / n_loops / n_threads  # make room for more lines

        data = {thread_id: {} for thread_id in thread_ids}
        events_to_analyze = [e for e in user_events]
        order = []

        function_elevations = {thread_id: {} for thread_id in thread_ids}

        while len(events_to_analyze) > 0:

            # use type of first one, remove all others of that type, repeat until empty
            e = events_to_analyze[0]
            thread_id = e['thread_id']
            thread_index = [i for i in range(n_threads) if thread_ids[i] == thread_id][0]
            if e['type'] == 'function':
                function_elevations[thread_id][e['name']] = len(function_elevations[thread_id])
            events = [event for event in events_to_analyze if (e['thread_id'] == event['thread_id'] and
                                                               e['name'] == event['name'])]
            events_to_analyze = [event for event in events_to_analyze if not (e['thread_id'] == event['thread_id'] and
                                                                              e['name'] == event['name'])]

            order.append((thread_id, e['name']))
            data[thread_id][e['name']] = {'events': [],
                                          'loop_intervals': [],
                                          'plot_y_coords': [],
                                          'loop_marker_times': [],
                                          'durations': [],
                                          'fractions': [],
                                          'count': len(events),
                                          'type': e['type']}
            for event in events:
                li = event['loop_index']
                try:
                    loop_start = loop_start_times[loop_reverse_index[li]]
                except:
                    import pprint
                    pprint.pprint(event)
                    import ipdb
                    ipdb.set_trace()
                loop_interval, duration, loop_time, fraction = None, None, None, None
                if event['type'] == 'function':
                    loop_interval = [event['start'] - loop_start, event['stop'] - loop_start]

                    duration = event['stop'] - event['start']
                    fraction = duration / loop_durations[loop_reverse_index[li]]
                elif event['type'] == 'marker':
                    loop_time = event['time'] - loop_start

                y_coord = loop_reverse_index[li] + thread_index * thread_spacing

                data[e['thread_id']][e['name']]['events'].append(event)
                data[e['thread_id']][e['name']]['loop_intervals'].append(loop_interval)
                data[e['thread_id']][e['name']]['plot_y_coords'].append(y_coord)
                data[e['thread_id']][e['name']]['loop_marker_times'].append(loop_time)
                data[e['thread_id']][e['name']]['durations'].append(duration)
                data[e['thread_id']][e['name']]['fractions'].append(fraction)
        max_funcs_per_thread = np.max([len([True for e_name in data[thread_id] if
                                            data[thread_id][e_name]['type'] == 'function']) for thread_id in
                                       thread_ids])
        elevation_scale = thread_spacing / max_funcs_per_thread

        if print_avgs:
            print("\n\nFunctions\tname\t\t\ttimes\t\tavg. duration (ms) [std.]\tavg duration (pct)")
            print("\n\t\t(all loops)\t\t%i\t%.3f (ms) [%.5f]" % (
                n_loops, np.mean(loop_durations) * 1000., np.std(loop_durations) * 1000))
            for thread_id in thread_ids:
                avg_fracs = []
                avg_durations = []
                thread_index = [i for i in range(n_threads) if thread_ids[i] == thread_id][0]
                print("\n\tThread:  %s%s" % (
                    thread_index, " (main)\n" if thread_id == LoopPerfTimer._MAIN_THREAD_ID else "\n"))
                for name in data[thread_id]:
                    count = data[thread_id][name]['count']
                    if data[thread_id][name]['type'] == 'function':
                        pct = np.mean(data[thread_id][name]['fractions'])
                        pct_str = "%.3f %%" % (pct * 100,)
                        dur = np.mean(data[thread_id][name]['durations'])
                        avg_fracs.append(pct)
                        avg_durations.append(dur)
                        dur_str = "%.6f (ms)" % (dur * 1000,)
                        dur_std_str = "[%.6f]" % (1000. * np.std(data[thread_id][name]['durations']))
                        print("\t\t%s\t\t%s\t%s %s\t\t%s" % (name, count, dur_str, dur_std_str, pct_str))

                total_frac = np.sum(avg_durations) / np.mean(loop_durations)
                print("\t\t(total)\t\t\t\t%.6f (ms)\t\t\t%.6f %%" % (np.sum(avg_durations) * 1000,
                                                                     100 * np.sum(avg_fracs)))
        t_max = 0.0

        if plot:
            n_colors = len(order) + 1
            colors = make_n_colors(n=n_colors)
            plot_handles = []
            plot_labels = []
            x_coords, y_coords, plot_kw, plot_str = [], [], {}, 'o'
            for pas in ['functions', 'markers']:
                for plot_ind, (thread_id, name) in enumerate(order):
                    thread_index = [i for i in range(n_threads) if thread_id == thread_ids[i]][0]
                    thread_label = "(main)" if thread_index == 0 else "(thread %i)" % (thread_index,)
                    count = data[thread_id][name]['count']
                    elev = function_elevations[thread_id][name] if name in function_elevations[thread_id] \
                        else thread_index * thread_spacing
                    thickness_mult = 1.0 if thread_index == 0 else 0.666
                    if data[thread_id][name]['type'] == 'function' and pas == 'functions':
                        x_coords = [x * 1000 for interval in data[thread_id][name]['loop_intervals'] for x in
                                    (interval + [np.nan])]
                        y_coords = [elev * elevation_scale + y for y in data[thread_id][name]['plot_y_coords'] for _ in
                                    range(3)]
                        plot_str = "-"
                        plot_kw = {'linewidth': thickness * thickness_mult, 'solid_capstyle': 'butt'}
                    elif data[thread_id][name]['type'] == 'marker' and pas == 'markers':
                        x_coords = [x * 1000 for x in data[thread_id][name]['loop_marker_times']]
                        y_coords = data[thread_id][name]['plot_y_coords']
                        plot_str = "o"
                        plot_kw = {'markersize': 6}
                    elif data[thread_id][name]['type'] not in ['function', 'marker']:
                        raise Exception("Unknown event type:  %s" % (data[thread_id][name]['type'],))
                    else:
                        continue
                    if len(x_coords) > 0:
                        t_max = np.nanmax([np.nanmax(x_coords), t_max])
                    plot_labels.append("%s%s (%s)" % (thread_label, name, count))
                    plot_handles.append(plt.plot(x_coords, y_coords, plot_str, color=colors[plot_ind], **plot_kw)[0])
            loop_end_x = loop_durations * 1000.
            t_max = np.nanmax([np.nanmax(loop_end_x), t_max])
            plot_handles.append(plt.plot(loop_end_x, range(n_loops), '.k', markersize=7)[0])
            plot_labels.append("loop ends")

            # plot lines separating loop iterations
            x_max = t_max * 1.025
            x_min = x_max * -0.025
            divisions_y = np.arange(0.0, n_loops)
            y_coords = [y for y in divisions_y for _ in range(3)]
            x_coords = [x for _ in divisions_y for x in (x_min, x_max, np.nan)]
            plt.plot(x_coords, y_coords, 'k-', linewidth=0.3)

            # plot lines for each thread
            thread_offsets = np.arange(1.0, n_threads) * thread_spacing
            y_coords = [yc + to for to in thread_offsets for yc in y_coords]
            x_coords = [xc for _ in thread_offsets for xc in x_coords]
            plt.plot(x_coords, y_coords, 'k:', linewidth=0.3)

            plt.title("Timing results for %i loops" % (n_loops,))
            plt.ylabel('loop index')
            plt.xlabel("ms")
            legend_names = [re.sub('^_', '', ln) for ln in plot_labels]
            plt.legend(plot_handles, legend_names, loc='upper right')
            # plt.gca().invert_yaxis()
            plt.xlim([x_min, x_max])
            plt.show()

    """
def make_n_colors(n, scale=(.8, .69, .46)):
    """
    Make a palette of evenly distanced colors.
    :param n:  how many to make?
    :param scale:  [0.0, 1.0] weights for R, G, and B
    :return:  n x 3 array of colors
    """
    color_range = np.linspace(0., np.pi, n + 1)

    colors = np.vstack([[scale[0] * np.abs(np.sin(color_range + np.pi / 4))],
                        [scale[1] * np.abs(np.sin(color_range + np.pi / 2.))],
                        [scale[2] * np.abs(np.sin(color_range))]])

    odds = colors[:, 1::2]
    colors[:, 1::2] = odds[::-1]
    colors = colors[:, :-1]
    return colors.T


def perf_sleep(t):
    start = time.perf_counter()
    while time.perf_counter() - start < t:
        pass
    return time.perf_counter() - start
