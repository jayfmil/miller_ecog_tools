# RAM_ECoG

Python toolbox for helping keeping your data and analyses organized. The currently implemented analyses are tailored towards electrocorticographic data collected for the Restoring Active Memory (RAM) project, but the code is easily exended to other types of data.

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


