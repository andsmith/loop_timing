import numpy as np
import matplotlib.pyplot as plt
import re
from events import EventTypes
import numpy as np
from util import uniquify


def print_profile_data(events, main_thread_id):
    pass


def _disambiguate_threads_and_functions(event_list, main_thread_id, thread_ids):
    """
    change function & marker names to include thread IDs.
        e.g. "function_1"  becomes "[1] function_1" if it's in the first thread.
        (no change for main)
    """
    # make sure main is first
    if main_thread_id in thread_ids:
        thread_ids = [thread_id for thread_id in thread_ids if thread_id != main_thread_id]
    thread_ids = [main_thread_id] + thread_ids
    thread_indices = {thread_id: index for index, thread_id in enumerate(thread_ids)}
    for e in event_list:
        if e['thread_id'] != main_thread_id:
            thread_index = thread_indices[e['thread_id']]
            if e['type'] in [EventTypes.MARKER, EventTypes.FUNC_CALL]:  # rename these
                e['name'] = "[%i] %s" % (thread_index, e['name'])
            elif e['type'] not in [EventTypes.LOOP_START]:  # don't rename these
                raise Exception("Unknown event type:  %s: " % (e['type'],))


def _get_function_name_order(events, function_names, n_threads):
    main_calls = [f for f in function_names if not f.startswith('[')]
    thread_calls = [[f for f in function_names if f.startswith('[%i]' % (i,))] for i in range(n_threads)]

    def _reorder(names):
        ordered_names = []
        index = 0
        while len(ordered_names) < len(names):
            if index >= len(events):
                raise Exception("function name not found in list of events:  %s" % (set(names) - set(ordered_names),))

            if 'name' in events[index] and events[index]['name'] in names:
                print(events[index])
                ordered_names.append(events[index]['name'])
            index += 1
        return ordered_names

    new_order = _reorder(main_calls)
    for i in range(n_threads):
        new_order += _reorder(thread_calls[i])
    return new_order


def plot_profile_data_old(events, main_thread_id):
    def _filter_sort(e_type, thread_id=None, loop_index=None):
        ev = [e for e in events if e['type'] == e_type]
        if thread_id is not None:
            ev = [e for e in ev if e['thread_id'] == thread_id]
        if loop_index is not None:
            ev = [e for e in ev if e['loop_index'] == loop_index]
        return sorted(ev, key=lambda x: x['time'])

    import pprint
    pprint.pprint(events)
    thread_ids = list(set([event['thread_id'] for event in events]))
    n_threads = len(thread_ids)

    _disambiguate_threads_and_functions(events, main_thread_id, thread_ids)

    loop_start_events = _filter_sort(EventTypes.LOOP_START)
    function_events = _filter_sort(EventTypes.FUNC_CALL)
    marker_events = _filter_sort(EventTypes.MARKER)

    marker_names = list(set([e['name'] for e in marker_events]))
    function_names = list(set([e['name'] for e in function_events]))
    function_names = _get_function_name_order(events, function_names, n_threads=n_threads)
    import pprint
    pprint.pprint(function_names)
    user_events = sorted(function_events + marker_events, key=lambda x: x['time'])

    loop_indices = sorted(list(set([e['loop_index'] for e in loop_start_events])))
    last_event_time = np.max([e['time'] for e in events])
    last_event_timex = last_event_time
    if len(function_events) > 0:
        last_event_time_alt = np.max([e['stop_t'] for e in function_events])
        last_event_time = np.max((last_event_time, last_event_time_alt))
    loop_start_times = [e['time'] for e in loop_start_events]
    loop_end_times = loop_start_times[1:] + [last_event_time]
    loop_durations = np.array(loop_end_times) - np.array(loop_start_times)

    non_main_thread_ids = [i for i in thread_ids if i != main_thread_id]
    n_loops = len(loop_start_events)

    if True:
        print("Collected data:")
        print("\tLoops:  %i" % (n_loops,))
        print("\tThreads:  %i" % (n_threads,))
        print("\tThread_ids:  %s" % (thread_ids,))

    plot_dims = {'loop_height': 1.0,
                 'thread_height': 0.7,
                 'loop_spacing': .05}

    # arrange where things will go vertically
    y = -plot_dims['loop_spacing']
    y_coords = {}

    events_sorted = {ind: {'main': {'func': [],
                                    'mark': []},
                           'threads': {}}
                     for ind in loop_indices}
    # events_sorted = {}
    heavy_lines_y = []
    light_lines_y = []

    for ind in loop_indices:

        y -= plot_dims['loop_spacing']
        heavy_lines_y.append(y)
        y -= plot_dims['loop_spacing']

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

                # if len(marks) > 0:
                events_sorted[ind]['threads'][t_id]['mark'] = marks

                # if len(funcs) > 0:
                events_sorted[ind]['threads'][t_id]['func'] = funcs

        # now calculate y-coords
        for t_id in non_main_thread_ids:
            if t_id not in events_sorted[ind]['threads']:
                continue
            light_lines_y.append(y)

            f_names = uniquify([e['name'] for e in events_sorted[ind]['threads'][t_id]['func']])
            m_names = uniquify([e['name'] for e in events_sorted[ind]['threads'][t_id]['mark']])

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

    heavy_lines_y.append(y - plot_dims['loop_spacing'])

    func_coords = {}
    mark_coords = {}

    line_styles = {}

    for ind in loop_indices:
        # main thread functions in this range
        funcs = events_sorted[ind]['main']['func']
        for e in funcs:
            if e['name'] not in func_coords:
                func_coords[e['name']] = []
            x_0 = e['start_t'] - loop_start_times[ind]
            x_1 = e['stop_t'] - loop_start_times[ind]
            y = y_coords[ind][main_thread_id]['func'][e['name']]
            line_styles[e['name']] = '-'
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
                line_styles[e['name']] = '-'
                func_coords[e['name']].append(np.array([[x_0, y], [x_1, y]]))
            marks = events_sorted[ind]['threads'][t_id]['mark']

            for e in marks:
                if e['name'] not in mark_coords:
                    mark_coords[e['name']] = []
                x = e['time'] - loop_start_times[ind]
                y = y_coords[ind][t_id]['mark'][e['name']]

                mark_coords[e['name']].append(np.array([x, y]))

    n_colors = len(func_coords)
    colors = make_n_colors(n=n_colors)
    plot_handles = []
    plot_labels = []
    x_coords, y_coords, plot_kw, plot_str = [], [], {}, 'o'

    # now plot everything

    # plot function calls
    for plot_ind, func_name in enumerate(function_names):
        if func_name not in func_coords:
            print("Weird:  %s" % (func_name,))
            continue

        coords = [np.vstack([coord, (np.nan, np.nan)]) for coord in func_coords[func_name]]

        coords = np.vstack(coords)
        plot_handles.append(
            plt.plot(coords[:, 0], coords[:, 1], line_styles[func_name], linewidth=3, color=colors[plot_ind], zorder=2,
                     **plot_kw)[0])
        plot_labels.append(func_name)

    # plot markers
    for plot_ind, mark_name in enumerate(marker_names):
        if mark_name not in mark_coords:
            print("Weird:  %s" % (mark_name,))
            continue

        coords = np.array(mark_coords[mark_name])

        plot_handles.append(plt.plot(coords[:, 0], coords[:, 1], "o", color=colors[plot_ind], zorder=3, **plot_kw)[0])
        plot_labels.append(mark_name)

    heavy_lines_y = np.array(heavy_lines_y)
    # plot divider lines
    xmin, xmax = plt.xlim()
    plt.hlines(heavy_lines_y[1:-1], xmin, xmax, linestyles='solid', colors='black', zorder=1, linewidth=.5)
    # plt.hlines(light_lines_y, xmin, xmax / 10, linestyles='dotted', colors='black', zorder=0, linewidth=0)
    plt.xlim(xmin, xmax)
    y_ticks = (heavy_lines_y[1:] + heavy_lines_y[:-1]) / 2.0

    plot_labels = [" %s" % (lab,) if lab.startswith('_') else lab for lab in plot_labels]
    plt.title("Timing results for %i loops" % (n_loops,))
    plt.ylabel('loop index')
    plt.yticks(y_ticks, loop_indices)
    plt.xlabel("ms")
    plt.legend(plot_handles, plot_labels, loc='upper right', title="[thread #] function/marker")
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
