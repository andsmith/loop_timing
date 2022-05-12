import numpy as np
import matplotlib.pyplot as plt
import re
from events import EventTypes
import numpy as np
from util import uniquify


def print_profile_data(events, main_thread_id):
    pass


def plot_profile_data(events, main_thread_id):
    def _filter_sort(e_type, thread_id=None, loop_index=None):
        ev = [e for e in events if e['type'] == e_type]
        if thread_id is not None:
            ev = [e for e in ev if e['thread_id'] == thread_id]
        if loop_index is not None:
            ev = [e for e in ev if e['loop_index'] == loop_index]
        return sorted(ev, key=lambda x: x['time'])

    loop_start_events = _filter_sort(EventTypes.LOOP_START)
    function_events = _filter_sort(EventTypes.FUNC_CALL)
    marker_events = _filter_sort(EventTypes.MARKER)
    marker_names = list(set([e['name'] for e in marker_events]))
    function_names = list(set([e['name'] for e in function_events]))
    user_events = sorted(function_events + marker_events, key=lambda x: x['time'])

    loop_indices = sorted(list(set([e['loop_index'] for e in loop_start_events])))
    last_event_time = np.max([e['time'] for e in events])
    last_event_timex = last_event_time
    last_event_time_alt = np.max([e['stop_t'] for e in function_events])
    last_event_time = np.max((last_event_time, last_event_time_alt))
    loop_start_times = [e['time'] for e in loop_start_events]
    loop_end_times = loop_start_times[1:] + [last_event_time]
    loop_durations = np.array(loop_end_times) - np.array(loop_start_times)

    thread_ids = list(set([event['thread_id'] for event in events]))
    non_main_thread_ids = [i for i in thread_ids if i != main_thread_id]
    n_threads = len(thread_ids)
    n_loops = len(loop_start_events)

    if True:
        print("Collected data:")
        print("\tLoops:  %i" % (n_loops,))
        print("\tThreads:  %i" % (n_threads,))
        print("\tThread_ids:  %s" % (thread_ids,))

    plot_dims = {'loop_height': 1.0,
                 'thread_height': 0.7}

    # arrange where things will go vertically
    y = 0
    y_coords = {}

    events_sorted = {ind: {'main': {'func': [],
                                    'mark': []},
                           'threads': {}}
                     for ind in loop_indices}
    # events_sorted = {}

    for ind in loop_indices:
        # main thread functions in this range
        y_inc = y - plot_dims['loop_height']
        y_coords[ind] = {main_thread_id: {'top': y, 'bottom': y_inc}}
        y = y_inc

        # how many function calls & markers are there
        events_sorted[ind]['main']['func'] = _filter_sort(EventTypes.FUNC_CALL, thread_id=main_thread_id,
                                                          loop_index=ind)
        events_sorted[ind]['main']['mark'] = _filter_sort(EventTypes.MARKER, thread_id=main_thread_id, loop_index=ind)
        f_names = uniquify([e['name'] for e in events_sorted[ind]['main']['func']])
        m_names = uniquify([e['name'] for e in events_sorted[ind]['main']['mark']])
        n_funcs = len(events_sorted[ind]['main']['func'])

        # where they go
        func_y_coords = np.linspace(y_coords[ind][main_thread_id]['bottom'],
                                    y_coords[ind][main_thread_id]['top'],
                                    n_funcs)
        mark_y_coord = (y_coords[ind][main_thread_id]['bottom'] +
                        y_coords[ind][main_thread_id]['top']) / 2

        y_coords[ind][main_thread_id]['func'] = {f_name: func_y_coords[i] for i, f_name in enumerate(f_names)}
        y_coords[ind][main_thread_id]['mark'] = {m_name: mark_y_coord for i, m_name in enumerate(m_names)}

        # Do it again for each thread (but differently)
        # first sort
        for t_id in non_main_thread_ids:
            funcs = _filter_sort(EventTypes.FUNC_CALL,
                                 thread_id=t_id,
                                 loop_index=ind)
            marks = _filter_sort(EventTypes.MARKER,
                                 thread_id=t_id,
                                 loop_index=ind)

            if len(funcs) > 0 or len(marks) > 0:
                events_sorted[ind]['threads'][t_id] = {}

                if len(marks) > 0:
                    events_sorted[ind]['threads'][t_id]['mark'] = marks

                if len(funcs) > 0:
                    events_sorted[ind]['threads'][t_id]['func'] = funcs

        # now calculate y-coords
        for t_id in non_main_thread_ids:
            if t_id not in events_sorted[ind]['threads']:
                continue
            f_names = uniquify([e['name'] for e in events_sorted[ind]['threads'][t_id]['func'] if 'func' in
                                events_sorted[ind]['threads'][t_id]])
            m_names = uniquify([e['name'] for e in events_sorted[ind]['threads'][t_id]['mark'] if 'mark' in
                                events_sorted[ind]['threads'][t_id]])

            n_funcs = len(events_sorted[ind]['main']['func'])

            y_inc = y - plot_dims['thread_height']
            y_coords[ind][t_id] = {'top': y,
                                   'bottom': y_inc}
            y = y_inc

            func_y_coords = np.linspace(y_coords[ind][t_id]['bottom'],
                                        y_coords[ind][t_id]['top'],
                                        n_funcs)

            mark_y_coord = (y_coords[ind][t_id]['bottom'] +
                            y_coords[ind][t_id]['top']) / 2

            y_coords[ind][t_id]['func'] = {f_name: func_y_coords[i] for i, f_name in enumerate(f_names)}

            y_coords[ind][t_id]['mark'] = {m_name: mark_y_coord for i, m_name in enumerate(m_names)}

    func_coords = {}
    mark_coords = {}

    for ind in loop_indices:
        # main thread functions in this range
        funcs = events_sorted[ind]['main']['func']
        for e in funcs:
            if e['name'] not in func_coords:
                func_coords[e['name']] = []
            x_0 = e['start_t'] - loop_start_times[ind]
            x_1 = e['stop_t'] - loop_start_times[ind]
            y = y_coords[ind][main_thread_id]['func'][e['name']]

            func_coords[e['name']].append(np.array([[x_0, y], [x_1, y]]))
        marks = events_sorted[ind]['main']['mark']

        for e in marks:
            if e['name'] not in mark_coords:
                mark_coords[e['name']] = []
            x = e['time'] - loop_start_times[ind]
            y = y_coords[ind][main_thread_id]['mark'][e['name']]

            mark_coords[e['name']].append(np.array([x, y]))

        for t_id in events_sorted[ind]['threads']:

            funcs = events_sorted[ind]['threads'][t_id]['func']
            for e in funcs:
                if e['name'] not in func_coords:
                    func_coords[e['name']] = []
                x_0 = e['start_t'] - loop_start_times[ind]
                x_1 = e['stop_t'] - loop_start_times[ind]
                y = y_coords[ind][t_id]['func'][e['name']]

                func_coords[e['name']].append(np.array([[x_0, y], [x_1, y]]))
            marks = events_sorted[ind]['threads'][t_id]['mark']

            for e in marks:
                if e['name'] not in mark_coords:
                    mark_coords[e['name']] = []
                x = e['time'] - loop_start_times[ind]
                y = y_coords[ind][t_id]['mark'][e['name']]

                mark_coords[e['name']].append(np.array([x, y]))

    import pprint
    pprint.pprint(func_coords)
    pprint.pprint(mark_coords)

    n_colors = len(func_coords)
    colors = make_n_colors(n=n_colors)
    plot_handles = []
    plot_labels = []
    x_coords, y_coords, plot_kw, plot_str = [], [], {}, 'o'

    

    for plot_ind, func_name in enumerate(function_names):
        if func_name not in func_coords:
            print("Weird:  %s" % (func_name,))
            continue

        coords = [np.vstack([coord, (np.nan, np.nan)]) for coord in func_coords[func_name]]
        pprint.pprint(coords)
        coords = np.vstack(coords)
        plot_handles.append(plt.plot(coords[:, 0], coords[:, 1], "-", color=colors[plot_ind], **plot_kw)[0])
        plot_labels.append(func_name)

    for plot_ind, mark_name in enumerate(marker_names):
        if mark_name not in mark_coords:
            print("Weird:  %s" % (mark_name,))
            continue

        coords = np.array(mark_coords[mark_name])

        plot_handles.append(plt.plot(coords[:, 0], coords[:, 1], "o", color=colors[plot_ind], **plot_kw)[0])
        plot_labels.append(mark_name)

    plt.title("Timing results for %i loops" % (n_loops,))
    plt.ylabel('loop index')
    plt.xlabel("ms")
    plt.legend(plot_handles, plot_labels, loc='upper right')
    # plt.gca().invert_yaxis()
    plt.show()


def make_n_colors(n, scale=(.8, .69, .46)):
    """
    Make   a palette of  evenly  distanced colors.
    :param n: how many to  make?
    :param scale: [0.0, 1.0]
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
