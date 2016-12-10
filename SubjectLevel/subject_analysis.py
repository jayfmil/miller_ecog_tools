import os
import cPickle as pickle
from SubjectLevel.subject_data import SubjectData


class SubjectAnalysis(SubjectData):

    def __init__(self, task=None, subject=None):
        super(SubjectAnalysis, self).__init__(task=task, subject=subject)
        self.load_res_if_file_exists = False
        self.save_res = True
        self.res_save_dir = None
        self.res_save_file = None
        self.res = {}

    def make_res_dir(self):
        """
        Create directory where results data will be saved/loaded if it needs to be created. This also will define
        self.res_save_dir and self.res_save_file
        """

        self.res_save_dir = self._generate_res_save_path()
        self.res_save_file = os.path.join(self.res_save_dir, self.subj + '_' + self.feat_type + '_' + self.res_str)
        if not os.path.exists(self.res_save_dir):
            try:
                os.makedirs(self.res_save_dir)
            except OSError:
                pass

    def load_res_data(self):
        """
        Load results if they exist and modify self.res to hold them.
        """
        if self.res_save_file is None:
            print('self.res_save_file must be defined before loading, .make_res_dir() will do this and create the '
                  'save directory for you.')
            return

        if os.path.exists(self.res_save_file):
            with open(self.res_save_file, 'rb') as f:
                res = pickle.load(f)
            self.res = res
        else:
            print('%s: No results to load.' % self.subj)

    def save_res_data(self):
        """

        """
        if not self.res:
            print('Results be loaded or computed before saving. Use .load_data() or .analysis()')
            return

        # write pickle file
        with open(self.res_save_file, 'wb') as f:
            pickle.dump(self.res, f, protocol=-1)

    def _generate_res_save_path(self):
        pass