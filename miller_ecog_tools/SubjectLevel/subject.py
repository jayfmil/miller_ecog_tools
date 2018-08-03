from miller_ecog_tools.SubjectLevel import Analyses


class Subject(object):
    """
    Base class upon which data and Analyses are built. A Subject has an associated task, subject code, and montage.

    In order to actually DO anything, you need to set the analyses attribute. See list of possible analyses with
    .list_possible_analyses()
    """

    def __init__(self, task=None, subject=None, montage=0, analysis_name=None):

        # set the attributes
        self.task = task
        self.subject = subject
        self.montage = montage

        # the setter will make self.analysis the analysis class. Now you don't have to import the specific
        # analysis module directly.
        self.analysis = None
        self.analysis_name = analysis_name

    def _construct_analysis(self, analysis_name):
        return Analyses.analysis_dict[analysis_name](self.task, self.subject, self.montage)

    @staticmethod
    def list_possible_analyses():
        for this_ana in Analyses.analysis_dict.keys():
            print('{}\n{}'.format(this_ana, Analyses.analysis_dict[this_ana].__doc__))

    @property
    def analysis_name(self):
        return self._analysis_name

    @analysis_name.setter
    def analysis_name(self, a):
        if a is not None:
            self.analysis = self._construct_analysis(a)
            self._analysis_name = a
        else:
            self.analysis = None
            self._analysis_name = None