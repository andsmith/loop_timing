from loop_profiler import LoopPerfTimer as lt
import numpy as np
from threading import Thread
import time


class LoopPerfTimerDemo(object):
    """Sample class to demonstrate LoopPerfTimer"""

    def __init__(self, n=25, burn=5):
        self._n = n
        self._burn_in = burn
        self._stop = False
        self._helper = HelperClass()
        self._thread = Thread(target=self._thread_proc)
        #self._thread.start()

    def stop(self):
        time.sleep(.1)  # wait for thread methods to finish
        self._stop = True

    @lt.time_function
    def _thread_method(self, x):

        val= x * np.mean(np.random.randn(700))
        print("Val:  %s" %(val,))
        return val

    def _thread_proc(self):
        a = 0
        while not self._stop:
            #print("a", a)
            a += 1
            lt.add_marker("test_mark_thread")
            a = self._thread_method(a)
            time.sleep(0.005)

    @lt.time_function
    def calculate_1(self, x):
        for _ in range(100):
            x += np.sum(np.random.rand(100))
        return x

    @lt.time_function
    def calculate_2(self, x):
        for _ in range(np.random.randint(50, 60, 1)[0]):
            x += np.sum(np.random.rand(1000))

        a = sub_calc_1(x)

        for _ in range(np.random.randint(100, 200, 1)[0]):
            x += np.sum(np.random.rand(1000))

        return x + x

    def run(self):
        a, b = 0, 0

        lt.reset(enable=True, burn_in=self._burn_in)
        for i in range(self._n):
            lt.mark_loop_start()
            self._helper.calculate_1()
            a = self.calculate_2(a)
            a = self.calculate_1(a)
            self._helper.foo_bar()
            b = self.calculate_2(a)
            lt.add_marker('test_mark_main')
            a = sub_calc_2(b)
        lt.disable()
        return a, a


@lt.time_function
def sub_calc_1(a):
    # spend a fixed amount of time
    return np.mean(np.random.randn(100000)) * a


@lt.time_function
def sub_calc_2(b):
    # spend more random time
    return np.mean(np.random.randn(30000)) * b


class HelperClass(object):
    def __init__(self, x=100.3):
        self._x = x

    @lt.time_function
    def calculate_1(self):
        size = np.random.randint(43, 47)
        self._x = np.linalg.det(np.random.randn((size * size)).reshape(size, size))
        return self._x

    @lt.time_function
    def foo_bar(self):
        size = np.random.randint(40, 45)
        self._x = np.linalg.det(np.random.randn((size * size)).reshape(size, size))
        return self._x


if __name__ == "__main__":
    lp_demo = LoopPerfTimerDemo(15, 5)
    lp_demo.run()
    lp_demo.stop()

    lt.display_data()
