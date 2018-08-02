from miller_ecog_tools.SubjectLevel import Analyses


class Subject(object):
    """
    Base class upon which data and Analyses are built. A Subject has an associated task, subject code, and montage.

    In order to actually DO anything, you need to set the analyses attribute. See list of possible analyses with
    .list_possible_analyses()
    """

    def __init__(self, task=None, subject=None, montage=0, analysis=None):

        # set the attributes
        self.task = task
        self.subject = subject
        self.montage = montage

        # the setter will make self.analysis the analysis class. Neat.
        self.analysis = analysis

    def _construct_analysis(self, analysis):
        return Analyses.analysis_dict[analysis](self.task, self.subject, self.montage)

    @staticmethod
    def list_possible_analyses():
        for this_ana in Analyses.analysis_dict.keys():
            print('{}\n{}'.format(this_ana, Analyses.analysis_dict[this_ana].__doc__))

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    def analysis(self, a):
        if a is not None:
            self._analysis = self._construct_analysis(a)
        else:
            self._analysis = None