"""
Base class for analyses. Methods for saving and loading results, and convenient run() function.
"""

import joblib
import os
from miller_ecog_tools.SubjectLevel.subject_data import SubjectData


class SubjectAnalysisBase(SubjectData):
    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectAnalysisBase, self).__init__(task=task, subject=subject, montage=montage)

    def run(self):
        """
        Convenience function to run analysis steps.

        1. Load data or compute
        2. Create results directory if needed
        3. Load results or compute
        4. Save results if desired.
        """

        # Step 1: load data
        if self.subject_data is None:
            self.load_data()

        # Step 2: create (if needed) directory to save/load
        self._make_res_dir()

        # Step 3: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 4: if not loaded ...
        if not self.res:

            # Step 4A: compute subsequenct memory effect at each electrode
            print('%s: Running.' % self.subject)
            self.analysis()

            # save to disk
            if self.save_res:
                self.save_res_data()

    def analysis(self):
        """
        This should be overridden in the analysis subclass.

        This should set self.res.
        """
        pass

    def load_res_data(self):
        """
        Load results if they exist and modify self.res to hold them.
        """
        if self.res_save_file is None:
            self._make_res_dir()

        if os.path.exists(self.res_save_file):
            print('%s: loading results.' % self.subject)
            self.res = joblib.load(self.res_save_file)
        else:
            print('%s: No results to load.' % self.subject)

    def save_res_data(self):
        """
        Save the pickle file that holds self.res.
        """
        if self.res_save_file is None:
            self._make_res_dir()

        if not self.res:
            print('%s: Results be loaded or computed before saving. Use .load_res_data() or .analysis()' % self.subject)
            return

        # write pickle file
        joblib.dump(self.res, self.res_save_file)

    def _generate_res_save_path(self):
        """
        This should be overridden by an an analysis subclass. Should return a path the the directory where results
        should be saved.

        I would suggest os.path.join(os.path.split(self.save_dir)[0], self.__class__.__name__+'_res') from within
        the analysis class. This will use a directory at the same level as the data directory, named after the analysis
        class.
        """
        pass

    def _make_res_dir(self):
        """
        Create directory where results data will be saved/loaded if it needs to be created. This also will define
        self.res_save_dir and self.res_save_file
        """
        if not self.res_str:
            print('%s: .res_str must be defined.')
            return

        self.res_save_dir = self._generate_res_save_path()
        self.res_save_file = os.path.join(self.res_save_dir, self.subject + '_' + self.res_str)
        if not os.path.exists(self.res_save_dir):
            try:
                os.makedirs(self.res_save_dir)
            except OSError:
                pass