# RAM_ECoG

My toolbox for analyzing electrocorticographic (ECoG) data collected for the Restoring Active Memory (RAM) project. Some of the code is specific to the way the data are stored and organized for RAM, but other parts may be a general interest. I don't expect (nor do I necessarily intend) that anybody other than myself actually use this, but if you are interested please note that this is a relatively new and ongoing project with frequent updates.

## Code Structure

There are two main levels at which data are processed: *SubjectLevel* and *GroupLevel*.

### Subject Level

#### Subject and SubjectData
The core object upon which everything else is built is the `Subject`. The `Subject` class has two required attributes, `.task` and `.subject`.

`task` must be a valid experiment run in the RAM project. Valid tasks currently: `['RAM_TH1', 'RAM_TH3', 'RAM_YC1', 'RAM_YC2', 'RAM_FR1', 'RAM_FR2', 'RAM_FR3']`. `subject` must the subject code of a subject who ran in a given `task`. If a valid task is entered with an invalid (or no) subject code, the list of valid subjects will be returned.

```
from SubjectLevel.subject import Subject

# Try to initialize a Subject with no subject code.
subj = Subject(task='RAM_TH1')
Invalid subject for RAM_TH1, must be one of R1076D, R1124J, R1132C, ...

# Now try with a correct subject code.
subj = Subject(task='RAM_TH1', subject='R1076D')
```

On its own, an instance of a `Subject` is not very useful. A subject is useful once it has data. Data in this case are ECoG recordings: multichannel brain recordings of voltage fluctuations that, in my case, I generally transform into the power (amplitude squared) of the voltage trace at a number of frequencies. We do this with the class ``SubjectData``.

```
from SubjectLevel.subject_data import SubjectData

# Let's create our subject directly from the SubjectData class
subj = SubjectData(task='RAM_TH1', subject='R1076D')
```

Now we have access to methods for loading data, and we can set attributes that determine exactly how the data are computed. We can save this data to disk too, in an auto-generated location based upon the current settings. This helps keeps things organized, but note that the file locations only make sense for the RAM project and the server this code runs on.
```
# To load voltage and compute power with the default settings, simply:
subj.load_data()

# to change the frequencies at which power is computed, for example 50 log-spaced frequencies between 1 and 200 Hz.
import numpy as np
subj.freqs = np.logspace(np.log10(1), np.log10(200), 50)
subj.load_data()

# save data
subj.save_data()
```

#### SubjectAnalysis
Now that we have a subject with data, we can do an analysis. The base class for this is ``SubjectAnalysis``. Like ``SubjectData``, ``SubjectAnalysis`` takes in task and subject arguments, but now we have access to general methods for filtering the data into experimental phases of interest and to loading and saving analysis results (if they exist) in an organized way on disk. What we need still need is to write actual analysis code that builds off of this design. Analysis specific code is stored in the  ``SubjectLevel.Analyses`` directory.

Custom analyses classes inherit from ``SubjectAnalysis`` and must have a ``run()`` method and an ``analysis()`` method. ``run()`` will load subject data if it is not already loaded, creates the directory to save the results (if needed), and will call ``analysis()``, which does the heavy lifting for the specific analysis.

A relatively simply analysis the Subsequent Memory Effect (SME), which compares, for each electrode and frequency, trials where memory was good to trials where memory was bad.

```
from SubjectLevel.Analyses.subject_SME import SubjectSME

# SubjectSME is a subclass of SubjectAnalysis
subj = SubjectSME(task='RAM_TH1', subject='R1076D')
subj.run()
```

After ``analysis()`` is complete, ``subj`` will have an attribute ``.res``, which is a dictionary of any relevant results that we wish to store or save. ``SubjectAnalysis`` subclasses frequently contain plotting code so we can visualize results quickly. For example, to visualize the SME for a given electrode, we can call:

```
# just enter an electrode number
subj.plot_spectra_average(elec=2)
```


why isn't this working
![Power Spectra](images/example_power_spect.pdf?raw=true "Example SME")
![Power Spectra](images/example_power_spect.pdf?raw=true "Example SME")