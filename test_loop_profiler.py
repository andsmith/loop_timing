from loop_profiler import LoopPerfTimer  as lt
import numpy as np
from threading import Thread
import time

class HelperClass(object):
    def __init__(self, x=100.3):
        self._x=x

    @lt.time_function
    def calculate_1(self):
        size = np.random.randint(20, 50)
        self._x = np.linalg.det(np.random.randn((size, size)))
        return self._x


    @lt.time_function
    def _blah_blah_B(self):
        size = np.random.randint(2, 5)
        self._x = np.linalg.det(np.random.randn((size, size)))
        return self._x


class LoopPerfTimerTester(object):
    """Sample class to demonstrate LoopPerfTimer"""

    def __init__(self, n=50):
        self._n = n
        self._stop = False
        self._helper = HelperClass()
        self._thread = Thread(target=self._thread_proc)
        self._thread.start()

    def stop(self):
        time.sleep(.1)  # wait for thread methods to finish
        self._stop = True

    @lt.time_function
    def _thread_method(self, x):
        return x * np.mean(np.random.randn(2300))

    def _thread_proc(self):
        a = 0
        while not self._stop:
            a += 1
            lt.add_marker("test_mark_thread")
            a = self._thread_method(a)
            time.sleep(0.001)

    @lt.time_function
    def calculate_1(self, x):
        for _ in range(100):
            x += np.sum(np.random.rand(100))
        return x

    @lt.time_function
    def calculate_2(self, x):
        for _ in range(np.random.randint(10, 100, 1)[0]):
            x += np.sum(np.random.rand(1000))

        a = sub_calc_1(x)
        return x + x

    def run(self):
        a, b = 0, 0

        lt.enable()
        for i in range(self._n):
            lt.mark_loop_start()

            a = self.calculate_2(a)
            #a = self.calculate_1(a)
            #b = self.calculate_2(a)
            # LoopPerfTimer.add_marker('test_mark_main')
            #a = sub_calc_2(b)
        lt.disable()
        return a, a


@lt.time_function
def sub_calc_1(a):
    # spend a random amount of time
    return np.mean(np.random.randn(100000)) * a


@lt.time_function
def sub_calc_2(b):
    # spend more random time
    return np.mean(np.random.randn(30000)) * b


if __name__ == "__main__":
    l = LoopPerfTimerTester(1)
    l.run()
    l.stop()

    lt.display_data(print_avgs=False, plot=True)
