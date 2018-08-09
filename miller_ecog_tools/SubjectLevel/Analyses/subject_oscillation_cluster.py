import os

from miller_ecog_tools.SubjectLevel.subject_analysis import SubjectAnalysisBase
from miller_ecog_tools.SubjectLevel.subject_eeg_data import SubjectEEGData


class SubjectOscillationClusterAnalysis(SubjectAnalysisBase, SubjectEEGData):
    """
    Subclass of SubjectAnalysis and SubjectEEGData that does ........
    """

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectOscillationClusterAnalysis, self).__init__(task=task, subject=subject, montage=montage)

        # string to use when saving results files
        self.res_str = 'new_save_string.p'

        # create any other analysis specific attributes here

    def _generate_res_save_path(self):
        self.res_save_dir = os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res')

    def analysis(self):
        """
        Do some analysis with the data in self.subject_data and put the results in the self.res dictionary.
        """
        if self.subject_data is None:
            print('%s: compute or load data first with .load_data()!' % self.subject)
