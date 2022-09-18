import numpy as np
import re
from loop_timing.events import EventTypes
import numpy as np

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


def _disambiguate_threads_and_functions(event_list, main_thread_id):
    """
    change function & marker names to include thread IDs.
        e.g. "function_1"  becomes "[1] function_1" if it's in the first thread.
        (no change for main)
    :param event_list: all events
    :param main_thread_id: id of main caller
    """
    thread_ids = list(set([event['thread_id'] for event in event_list]))

    # make sure main is first
    if main_thread_id in thread_ids:
        thread_ids = [thread_id for thread_id in thread_ids if thread_id != main_thread_id]

    thread_indices = {thread_id: index for index, thread_id in enumerate(thread_ids)}

    for e in event_list:
        if e['thread_id'] != main_thread_id:
            thread_index = thread_indices[e['thread_id']]
            if e['type'] in [EventTypes.MARKER, EventTypes.FUNC_CALL]:  # rename these
                e['name'] = "[%i] %s" % (thread_index, e['name'])
            elif e['type'] not in [EventTypes.LOOP_START]:  # don't rename these
                raise Exception("Unknown event type:  %s: " % (e['type'],))


def _get_name_order(events, names):
    """
    Determine the order of the legend items.
    :param events:  all events
    :names:  the list of names to order
    :returns:  permutation of names, such that:
        Main thread functions/markers come before those of other threads.
        Within each thread, the order is the same as the order they were first called (on the first loop).
    """
    mains = [f for f in names if not f.startswith('[')]
    threads = [f for f in names if f.startswith('[')]
    thread_prefixes = list(set([re.search(r'^(\[[0-9]+\])', tc).groups()[0] for tc in threads]))
    threads_sorted = [[tc for tc in threads if tc.startswith(prefix)] for prefix in thread_prefixes]

    def _reorder(name_group):
        """
        Re-order a subset of names
        :param name_group:  rearrange so in the same order as they first appear in the event list
        """
        ordered_names = []
        index = 0
        while len(ordered_names) < len(name_group):
            if index >= len(events):
                raise Exception("function names not found in list of events:  %s" % (set(names) - set(ordered_names),))
            if 'name' in events[index]:
                if events[index]['name'] in name_group:
                    if events[index]['name'] not in ordered_names:
                        ordered_names.append(events[index]['name'])
            index += 1
        return ordered_names

    new_order = _reorder(mains)

    for thread_call_group in threads_sorted:
        new_order += _reorder(thread_call_group)

    return new_order


def plot_profile_data(events, main_thread_id, chop_early=False, burn_in=0):
    """
    Plot data, after profiler has been deactivated
    :param events:  list of all events
    :main_thread_id:  which events, etc
    :chop_early:  Remove events appearing before first loop_start
    :burn_in:  Discard this many loops before collecting data.
    """
    # need this here to avoid problems with cv2
    import matplotlib.pyplot as plt

    events = sorted(events, key=lambda event: event['time'])
    if not chop_early:
        for e in events:
            if e['loop_index'] < 0:
                e['loop_index'] = 0
    plot_dims = {'spacing': {'separator': 0.1,
                             'function': .15,
                             'marker': 0.1},
                 'function_width_base': 4,  # reduce if plotting lots of things
                 'separator_spacing': 0.05,
                 'separator_width': 0.5,
                 'separator_color': 'gray',
                 'marker_size': 3,
                 'marker': 'o'}

    def _filter_sort(**constraints):
        """
        Get a list of events satisfying constraints and sorted by 'time' key
        :param constraints:  all args are dict keys to events,
        :returns:  events matching value, in temporal order
        """
        ev = events
        for constraint_name in constraints:
            ev = [e for e in ev if constraint_name in e and e[constraint_name] == constraints[constraint_name]]
        return sorted(ev, key=lambda x: x['time'])

    thread_ids = list(set([event['thread_id'] for event in events]))
    n_threads = len(thread_ids)

    _disambiguate_threads_and_functions(events, main_thread_id)

    loop_start_events = _filter_sort(type=EventTypes.LOOP_START)
    function_events = _filter_sort(type=EventTypes.FUNC_CALL)
    marker_events = _filter_sort(type=EventTypes.MARKER)
    if len(loop_start_events) == 1:
        latest = np.max([e['time'] for e in events])
        function_events = _filter_sort(type=EventTypes.FUNC_CALL)
        last_function_end = np.max([e['stop_t'] for e in function_events])
        latest = np.max([latest, last_function_end])
        mean_loop_duration = latest - loop_start_events[0]['time']
    elif len(loop_start_events) > 1:
        mean_loop_duration = np.mean(np.diff([e['time'] for e in loop_start_events]))
    else:
        raise Exception("No Data!")

    time_scale = 1.0
    time_units = "seconds"
    if mean_loop_duration < 1.0:
        time_scale = 1000.0
        time_units = "miliseconds"
    if mean_loop_duration < .001:
        time_scale = 1000000.0
        time_units = "microseconds"

    mark_names = list(set([e['name'] for e in marker_events]))
    funct_names = list(set([e['name'] for e in function_events]))

    order = _get_name_order(events, funct_names + mark_names)

    loop_indices = sorted(list(set([e['loop_index'] for e in loop_start_events])))

    function_coords = {func_name: [] for func_name in funct_names}
    marker_coords = {marker_name: [] for marker_name in mark_names}
    name_types = {}
    y_val = [0]  # plot down, so first loop is on top
    last_plot_type = ['separator']
    n_items = [0]

    def _space(next_plot_type):
        """
        Determine how much vertical space to add, then do it.
        """
        n_items[0] += 1
        if next_plot_type == last_plot_type[0] == 'function':
            inc = plot_dims['spacing']['function']
        elif next_plot_type == 'separator' or last_plot_type[0] == 'separator':
            inc = plot_dims['spacing']['separator']
        else:
            inc = plot_dims['spacing']['marker']

        y_val[0] -= inc
        last_plot_type[0] = next_plot_type

    separator_coords = [y_val[0]]

    for i, loop_index in enumerate(loop_indices):
        loop_start_time = loop_start_events[i]['time']
        for name in order:

            event_subset = _filter_sort(loop_index=loop_index, name=name)
            if len(event_subset) == 0:
                continue
            name_types[event_subset[0]['name']] = event_subset[0]['type']
            if event_subset[0]['type'] == EventTypes.FUNC_CALL:
                _space('function')
                y_start, y_stop = y_val[0], y_val[0]
                for event in event_subset:
                    x_start = event['start_t'] - loop_start_time
                    x_stop = event['stop_t'] - loop_start_time
                    function_coords[name].extend([[x_start, y_start],
                                                  [x_stop, y_stop],
                                                  [np.NaN, np.NaN]])  # NaNs for plotting disjointed line segments
            elif event_subset[0]['type'] == EventTypes.MARKER:
                _space('marker')
                for event in event_subset:
                    x = event['time'] - loop_start_time
                    marker_coords[name].append([x, y_val[0]])
        _space('separator')
        separator_coords.append(y_val[0])

    plot_handles = []
    plot_labels = order
    colors = {order[i]: color for i, color in enumerate(make_n_colors(n=len(order)))}
    if n_items[0] < 50:
        plot_width = 6
        marker_size = 6
    elif n_items[0] < 250:
        plot_width = 4
        marker_size = 4
    else:
        plot_width = 2
        marker_size = 2

    for name in order:
        if name_types[name] == EventTypes.FUNC_CALL:
            coords = np.array(function_coords[name])
            coords[:, 0] *= time_scale
            plot_handles.append(
                plt.plot(coords[:, 0],
                         coords[:, 1],
                         '-',
                         linewidth=plot_width,
                         color=colors[name],
                         zorder=2,
                         solid_capstyle='butt')[0])
        elif name_types[name] == EventTypes.MARKER:
            coords = np.array(marker_coords[name])
            coords[:, 0] *= time_scale
            plot_handles.append(
                plt.plot(coords[:, 0],
                         coords[:, 1],
                         plot_dims['marker'],
                         markersize=marker_size,
                         color=colors[name],
                         zorder=3)[0])
    sc = np.array(separator_coords)
    tick_y_coords = (sc[1:] + sc[:-1]) / 2

    # plot divider lines
    x_min, x_max = plt.xlim()
    plt.hlines(separator_coords[1:-1], x_min, x_max, linestyles='dashed', colors='black', zorder=1, linewidth=.5)
    plt.xlim(x_min, x_max)
    plt.ylim(sc[-1], sc[0])
    plot_labels = [" %s" % (lab,) if lab.startswith('_') else lab for lab in plot_labels]
    plt.title("Timing results for %i loops  (%i skipped)" % (len(loop_indices), burn_in))
    plt.ylabel('loop index')
    plt.yticks(tick_y_coords, loop_indices)
    plt.xlabel(time_units)
    plt.legend(plot_handles, plot_labels, loc='upper right', title="[thread #] function/marker")
    # plt.gca().invert_yaxis()
    plt.show()
    return
