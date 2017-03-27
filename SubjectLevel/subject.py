import ram_data_helpers


class Subject(object):
    """
    Base class upon which data and Analyses are built. Defines subject name and experiment, and forces them to be valid.
    """
    valid_tasks = ['RAM_TH1', 'RAM_TH3', 'RAM_YC1', 'RAM_YC2', 'RAM_FR1', 'RAM_FR2', 'RAM_FR3', 'RAM_THR', 'RAM_PAL1']

    def __init__(self, task=None, subject=None, montage=0, use_json=True):

        # these are checked to be valid tasks and subjects
        self.use_json = use_json
        self.task = task
        self.subj = subject
        self.montage = montage

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, t):
        if t in self.valid_tasks:
            self._task = t
        else:
            self._task = None
            print('Invalid task, must be one of %s.' % ', '.join(self.valid_tasks))

    @property
    def subj(self):
        return self._subj

    @subj.setter
    def subj(self, s):
        if self.task is not None:
            valid_subjs = ram_data_helpers.get_subjs(self.task, self.use_json)
            if s in valid_subjs:
                self._subj = s
            else:
                self._subj = None
                print('Invalid subject for %s, must be one of %s.' % (self.task, ', '.join(valid_subjs)))
        else:
            print('Must set valid task.')
            self._subj = None
