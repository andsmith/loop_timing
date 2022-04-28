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

        y_coords[ind][main_thread_id]['func'] = [{'name': f_name,
                                                  'y_coord': func_y_coords[i]}
                                                 for i, f_name in enumerate(f_names)]
        y_coords[ind][main_thread_id]['mark'] = {'y_coord': mark_y_coord,
                                                 'names': m_names}

        # Do it again for each thread (but differently)
        # first sort
        for t_id in non_main_thread_ids:
            funcs = _filter_sort(EventTypes.FUNC_CALL,
                                 thread_id=t_id,
                                 loop_index=ind)
            marks = _filter_sort(EventTypes.MARKER,
                                 thread_id=t_id,
                                 loop_index=ind)

            print(funcs)
            if len(funcs) > 0 or len(marks)>0:
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

            y_coords[ind][t_id]['func'] = [{'name': f_name,
                                            'y_coord': func_y_coords[i]}
                                           for i, f_name in enumerate(f_names)]

            y_coords[ind][t_id]['mark'] = {'y_coord': mark_y_coord,
                                           'names': m_names}

    import pprint
    pprint.pprint(y_coords)


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
