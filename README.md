# loop_timing
Thread-aware profiling & performance analysis for apps with realtime loops.

## Usage
1. `from loop_timing.loop_profiler import LoopPerfTimer`
2. Decorate each function that is part of the main loop:
```
    @LoopPerfTimer.time_function()
    def calculate_1(self, x):
        ...
```
3. Add some function calls to the main loop:
```
   while True:
        LoopPerfTimer.mark_loop_start()
        ...
   LoopPerfTimer.mark_stop()
```
4. Call the display function to print/graph timing data:
```
    LoopPerfTimer.display_data(print_avgs=True, plot=True)
```
Test sample output:
```
        Functions name                  times   avg. duration (ms) [std.]       avg duration (pct)
                (all loops)             10      4.607 (ms) [0.65738]

        Thread:  0 (main)
                calculate_2             20      1.348680 (ms) [0.482694]        28.805 %
                sub_calc_1              20      0.336990 (ms) [0.013096]         7.462 %
                calculate_1             10      0.844430 (ms) [0.093979]        18.709 %
                sub_calc_2              10      1.048600 (ms) [0.201016]        23.322 %
                (total)                         3.578700 (ms)                   78.297769 %

        Thread:  1
                _thread_method          3       0.684300 (ms) [0.373257]        14.772 %
                (total)                         0.684300 (ms)                   14.772048 %
```
![Test sample output](https://github.com/andsmith/loop_timing/blob/main/sample_output.png)