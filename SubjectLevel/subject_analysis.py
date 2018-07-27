import os
import joblib
import numpy as np
from SubjectLevel.subject_data import SubjectData
import pdb


class SubjectAnalysis(SubjectData):
    """
    Main class for doing subject level analyses. Inherits from SubjectData. Contains methods for creating results
    directory, loading results, saving results, and a helper for filtering data to encoding or retrieval phases.

    Specific analyses should build off this class.
    """

    def __init__(self, task=None, subject=None, montage=0, use_json=True):
        super(SubjectAnalysis, self).__init__(task=task, subject=subject, montage=montage, use_json=use_json)
        self.load_res_if_file_exists = False
        self.save_res = True
        self.res_save_dir = None
        self.res_save_file = None
        self.verbose = False
        self.res = {}

        # this is generally defined by a subclass
        self.res_str = ''

    def filter_data_to_task_phases(self, phases=('enc', 'rec')):
        """
        Our tasks have both encoding ('enc') and retrieval ('rec') phases. This filters .subject_data to only include
        ['enc'] or ['rec'], or both ['enc', 'rec'] if desired. Modifies .subject_data in place.
        """
        if self.subject_data is None:
            print('%s: Data must be loaded before filtering to desired phases. Use .load_data()' % self.subj)
            return

        # create array of task phases for each event, accounting for the different strings for different experiments
        task_phase = self.subject_data.events.data['type']

        if 'RAM_YC' in self.task:
            enc_str = 'NAV_LEARN'
            rec_str = 'NAV_TEST'
        elif 'RAM_TH' in self.task:
            enc_str = 'CHEST'
            rec_str = 'REC'
            # if 'RAM_THR' in self.task:
            #     rec_str = 'REC_EVENT'
            #     rec_str = 'PROBE'
        elif 'RAM_PAL' in self.task:
            enc_str = 'STUDY_PAIR'
            rec_str = 'TEST_PROBE'
        else:
            enc_str = 'WORD'
            rec_str = 'REC_WORD'
        # enc_str = 'CHEST' if 'RAM_TH' in self.task else 'WORD'
        # rec_str = 'REC' if 'RAM_TH' in self.task else 'REC_WORD'
        task_phase[task_phase == enc_str] = 'enc'
        task_phase[task_phase == rec_str] = 'rec'

        # boolean of which events include
        phase_bool = np.array([True if x in phases else False for x in task_phase])

        # filter
        self.subject_data = self.subject_data[phase_bool]
        self.task_phase = self.task_phase[phase_bool]

    def make_res_dir(self):
        """
        Create directory where results data will be saved/loaded if it needs to be created. This also will define
        self.res_save_dir and self.res_save_file
        """
        if not self.res_str:
            print('%s: .res_str must be defined.')
            return

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
            self.make_res_dir()

        if os.path.exists(self.res_save_file):
            print('%s: loading results.' % self.subj)
            self.res = joblib.load(self.res_save_file)
        else:
            print('%s: No results to load.' % self.subj)

    def save_res_data(self):
        """
        Save the pickle file that holds self.res.
        """
        if self.res_save_file is None:
            self.make_res_dir()

        if not self.res:
            print('%s: Results be loaded or computed before saving. Use .load_res_data() or .analysis()' % self.subj)
            return

        # write pickle file
        joblib.dump(self.res, self.res_save_file)

    def _generate_res_save_path(self):
        """
        This should be overridden by a subclass.
        """
        pass
