# loop_timing
Thread-aware profiling & performance analysis for apps with realtime loops.

## Usage
1. Import the profiler:
```
from loop_timing.loop_profiler import LoopPerfTimer
```           
2. Don't instantiate, but decorate each function/method to be profiled:
```
    @LoopPerfTimer.time_function
    def calculate_1(self, x):
```
2. Add some markers you want to annotate the profile.
```
    ...
    LoopPerfTimer.add_marker("halfway")
    ...
```
3. Add some function calls to synchronize the profiler:
```
   lt.enable(burn_in=5)
   for iteration in range(15):
        LoopPerfTimer.mark_loop_start()
        ...
   LoopPerfTimer.mark_stop()
```
4. Call the display function to print/graph timing data:
```
    LoopPerfTimer.display_data()
```
Run the demo: `> python demo_loop_profiler.py`

![sample output](https://github.com/andsmith/loop_timing/blob/main/demo_output.png?raw=true)

[NOTE:  running on my i5-4250U CPU @ 1.30GHz]
## To do
Incorporate multiprocessing into profiler: 
```  
(process #) [thread #] Classname.methodname
