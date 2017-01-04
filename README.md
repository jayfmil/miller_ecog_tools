# RAM_ECoG

My toolbox for analyzing electrocorticographic (ECoG) data collected for the Restoring Active Memory (RAM) project. Some of the code is specific to the way the data are stored and organized for RAM, but other parts may be a general interest. I don't expect (nor do I necessarily intend) that anybody other than myself actually use this, but if you are interested please note that this is a relatively new and ongoing project with frequent updates.

## Code Structure

There are two main levels at which data are processed: *SubjectLevel* and *GroupLevel*.

### Subject Level Analyses

The core object upon which everything else is built is the `Subject`