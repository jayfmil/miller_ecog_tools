# RAM_ECoG

Python toolbox for helping keeping your data and analyses organized. The currently implemented analyses are tailored towards electrocorticographic data collected for the Restoring Active Memory (RAM) project, but the code is easily exended to other types of data.

*Why use this toolbox?*
A large challenge when analyzing data is simply keeping your data and results organized. This toolbox provides convenient methods for loading data, saving data, loading results, and saving results, and it uses automatically generated file paths baed on your analysis parameters to keep everything nicely organized on disk. It also provides a standardized pipeline analyzing an experiment, in which raw data for a subject is loaded, possibly transformed in some way and saved, a specific analysis is applied to those data, and the results are then saved.

## Code design

The fundamental unit of analysis is a *subject*, which is a single individual who participated in a specific experiment. When you create a *subject*, you also specific the specific analysis you are performing (in this case `SubjectSMEAnalysis`, a Subsequent Memory Analyis):

```python
from miller_ecog_tools.subject import create_subject
subject = create_subject(task='TH1', subject='R1289C', montage=0, analysis_name='SubjectSMEAnalysis')
```

Where `task` indicates the name of the experiment performance, `subject` is subject identifier code, and `montage` is the "montage number" for a subject's electrode configuration (montage may not be applicable to all experiments). `analysis_name` is the name of the analysis class you to instantiate. If you don't enter an analysis_name, a list of possible analyses will be printed. Note that `create_subject()` is really just a tool to make it easier to create your subject with specific data and analysis methods, you could do that same thing with:

```python
from miller_ecog_tools.SubjectLevel.Analyses.subject_SME import SubjectSMEAnalysis
subject = SubjectSMEAnalysis(task='TH1', subject='R1289C', montage=0)
```
`SubjectSMEAnalysis` is a subclass of the classes `SubjectEEGData` and `SubjectAnalysisBase`, and thus has access to their methods. `SubjectEEGData` is a subclass of `SubjectDataBase` that is specific to loading eeg data, and it has methods for loading, computing, and saving data. `SubjectAnalysisBase` has methods for loading, computing, and saving analysis results.

## Actually running an analysis

First, create a *subject*, as shown above. Then set the attributes of the data and analysis. In the case of analyses using `SubjectEEGData`, which loads eeg data and then performs spectral, you may set:

```python
# whether to load bipolar pairs of electrodes or monopolar contacts
self.bipolar = True

# This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
self.mono_avg_ref = False

# the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
# that will be applied to the events. Function must return events dataframe.
self.event_type = ['WORD']

# power computation settings
self.start_time = -500
self.end_time = 1500
self.wave_num = 5
self.buf_ms = 2000
self.noise_freq = [58., 62.]
self.resample_freq = None
self.log_power = True
self.freqs = np.logspace(np.log10(1), np.log10(200), 8)
self.mean_over_time = False
self.time_bins = None
self.use_mirror_buf = False
```
So if I wanted to compute power starting at 0 ms and continuing until 2000 ms (relative to each event in `.event_type`) and leave the other parameters the same, I would set:

```python
subject.start_time = 0 
subject.end_time = 2000
```
This would automatically set the save directory to the following path, based on the parameters:
```
/scratch/jfm2/python/TH1/8_freqs_1.000_200.000_bipol/0_start_2000_stop/1_bins/R1276D/0/power
```


## Code Structure

There are two main levels at which data are processed: *SubjectLevel* and *GroupLevel*.  *SubjectLevel* code deals with an individual subject only, and *GroupLevel* code helps with aggregating the results from multiple subjects.

## SubjectLevel - the SubjectDataBase class

When you create a subject (as shown below), you are really creating a subclass of the classes `SubjectDataBase` and `SubjectAnalysisBase`. `SubjectDataBase` has methods for loading, computing, and saving data:

## SubjectLevel - the SubjectAnalysisBase class

 `SubjectAnalysisBase` has methods for loading, computing, and saving analysis results.

## SubjectLevel - actually running an analysis

The fundamental unit of analysis is a `Subject`. A subject represents a single individual who participated in a specific experiment. A `Subject` can be created by:

```python
from miller_ecog_tools.subject import create_subject
subj = create_subject(task='TH1', subject='R1289C', montage=0, analysis_name='SubjectSMEAnalysis')
```

Where `task` indicates the name of the experiment performance, `subject` is subject identifier code, and `montage` is the "montage number" for a subject's electrode configuration (montage may not be applicable to all experiments). `analysis_name` is the name of the analysis class you to instantiate. If you don't enter an analysis_name, a list of possible analyses will be printed. Note that `create_subject()` is really just a tool to make it easier to create your subject with a specific analysis, you could do that same thing with:

```python
from miller_ecog_tools.SubjectLevel.Analyses.subject_SME import SubjectSMEAnalysis
subject = SubjectSMEAnalysis(task='TH1', subject='R1289C', montage=0)
```

but using `create_subject()` means you don't have to keep track of the file and class names yourself.

### setting data parameters


![Power Spectra](images/example_power_spect.png?raw=true)


