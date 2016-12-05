import TH_createPower
import numpy as np

def run(subjs, bipolar, freqs, window_size, step_size, do_par):
    th = TH_createPower.PowerCalc(subjs, bipolar, freqs, window_size, step_size, do_par)
    th.create_power_run()

if __name__ == '__main__':

    # If None, will do all subjs
    subjs = None

    # non changing inputs
    bipolar = True
    freqs = np.logspace(np.log10(1), np.log10(200), 8)
    #freqs = np.logspace(np.log10(1), np.log10(200), 50)
    step_size = 100.
    do_par = True
    #run(subjs, bipolar, freqs, 1000., step_size, do_par)

    # looping over different window sizes
    window_size_array = np.arange(100., 3900., 100.)
    for i, window_size in enumerate(window_size_array):
        print 'Creating power for window size %d (%d of %d)' % (window_size, i+1, len(window_size_array))
        run(subjs, bipolar, freqs, window_size, step_size, do_par)
