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
subj.plot_spectra_average(elec=5)
```

![Power Spectra](images/example_power_spect.png?raw=true)

Here, the top panel shows the power spectra (power spectral density) for a superiorfrontal electrode, with items that were later remembered in red and items that were later forgotten in blue. The bottom panel shows the resulting t-statistic from comparing the two distributions at each frequency. Shaded regions indicate significant differences at p<.05.

A more complicated analysis uses a logistic regression to actually predict whether individual trials will be encoded well (good memory) or poorly (bad memory):

```
from SubjectLevel.Analyses.subject_classifier import SubjectClassifier

# SubjectClassifier is another subclass of SubjectAnalysis.
# By default, it runs a L2 regularized logistic regression with leave-out-session-out cross validiation
# (or leave-out-trial-out if only one session of data).
subj = SubjectClassifier(task='RAM_TH1', subject='R1076D')
subj.run()
```

Again, we can quickly visualize the results, here classifier performance.

```
# Plot the reciever operator characteristic curve, showing how well the classifier can distingush the two classes of data.
subj.plot_roc()
```

![ROC](images/example_roc.png?raw=true)

### Group Level

#### Group

``Group()`` is the base class that handles performing an analysis across a set of subjects and collating the results. It is initialized as:

```
def __init__(self, analysis='classify_enc', subject_settings='default', open_pool=False, n_jobs=50, **kwargs):
```

`analysis` and `subject_settings` allow you to specify any arbitrary subject data parameters or analysis code that you wish to run. The file `GroupLevel.default_analyses` contains my mapping of `analysis` and `subject_settings` strings onto the actual settings. `open_pool` will open a parallel pool on our cluster to allow you to parallelize your code across a number of compute nodes. (number of nodes is `n_jobs`). You can also enter as keyword arguments any relevant attribute that you wish to set for your analysis. Any errors that occur will also automatically be logged using python's logging module.

The `.process` methods is called to begin the group analysis.
```
from GroupLevel.group import Group

# To run everything with completely default settings
group_res = Group()
group_res.process()
```

Similar to subject level analyses, specific group analyses should build off this class. These subclasses can have there own overriding `process()` method, but this then must call the parent `process()`. Doing this is a good way to have analysis specific code. For example, `GroupLevel.Analyses.group_classifier` will create a pandas dataframe `.summary_table` to summarize the group results. Subclasses are also a good place to put analysis specific plotting functions.

```
from GroupLevel.Analyses import group_classifier

# run the classifier with 50 frequencies
freqs = np.logspace(np.log10(1), np.log10(200), 50)
group_res = group_classifier.GroupClassifier(freqs=freqs)
group_res.process()
```

We can easily plot a histogram of classifier AUC values for the group, which will also tell us if our mean classification performance is above chance for the whole group.

```
group_res.plot_auc_hist()
```

![AUC Histogram](images/example_group_auc.png?raw=true)

Or we can plot the average importance of the classifier model weights, as measured by a forward model, as a function of brain region. In this case, this tells us which brain regions contribute most strongly to memory encoding. Washed out pixel are not significant at the group level, all others are.
```
# Visualize the classifier model as a function of brain reigon.
# FC: frontal cortex; Hipp: hippocampus; IFG: inferior frontal gyrus
# IPC: inferior parietal cortex; MFG: middle frontal gyrus; MTL: medial temporal lobe cortex
# OC: occipital cortex, SFG: superior frontal gyrus; SPC: superior parietal cortex
# TC: temporal cortex
#
IFG=inferior frontal gyrus; MFG=middle frontal gyrus; SFG=superior frontal gyrus; MTL=medial temporal lobe cortex; Hipp=hippocampus; TC=temporal cortex; IPC=inferior parietal cortex; SPC=superior parietal cortex; OC=occipital cortex.
group_res.plot_feature_map()
```
![Feature Map](images/example_feature_map.png?raw=true)