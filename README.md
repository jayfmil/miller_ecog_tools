# RAM_ECoG

My toolbox for analyzing electrocorticographic (ECoG) data collected for the Restoring Active Memory (RAM) project. Some of the code is specific to the way the data are stored and organized for RAM, but other parts may be a general interest. I don't expect (nor do I necessarily intend) that anybody other than myself actually use this, but if you are interested please note that this is a relatively new and ongoing project with frequent updates.

## Code Structure

There are two main levels at which data are processed: *SubjectLevel* and *GroupLevel*.

### Subject Level Analyses

The core object upon which everything else is built is the `Subject`. The `Subject` class has two required attributed, `.task` and `.subject`.

`task` must be a valid experiment run in the RAM project. Valid tasks currently: `['RAM_TH1', 'RAM_TH3', 'RAM_YC1', 'RAM_YC2', 'RAM_FR1', 'RAM_FR2', 'RAM_FR3']`. `subject` must the subject code of a subject who ran in a given `task`. If a valid task is entered with an invalid (or no) subject code, the list of valid subjects will be returned.

```
from SubjectLevel.subject import Subject

# Try to initialize a Subject with no subject code.
subj = Subject(task='RAM_TH1')
Invalid subject for RAM_TH1, must be one of R1076D, R1124J, R1132C, ...

# Now try with it correctly.
subj = Subject(task='RAM_TH1', subject='R1076D')
```