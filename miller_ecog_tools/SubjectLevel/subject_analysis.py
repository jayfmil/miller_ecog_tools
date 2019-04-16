"""
Base class for analyses. Methods for saving and loading results, and convenient run() function.
"""

import joblib
import os
from miller_ecog_tools.subject import SubjectDataBase


class SubjectAnalysisBase(SubjectDataBase):
    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectAnalysisBase, self).__init__(task=task, subject=subject, montage=montage)

        # settings for handling the loading/saving and computation of results
        self.load_res_if_file_exists = False  #
        self.save_res = True                  #
        self.auto_save_data = True            #
        self.res_save_dir = None              #
        self.res_save_file = None             #
        self.ana_requires_data = True         # If False, will not call .load_data() before running.
        self.do_not_compute_res = False       #
        self.force_analysis = False           # Will redo the analysis even if the res file exists
        self.verbose = False
        self.res = {}

        # this is generally defined by a subclass
        self.res_str = ''

    # automatically set .res_save_dir when .res_str is set
    @property
    def res_str(self):
        return self._res_str

    @res_str.setter
    def res_str(self, x):
        self._res_str = x
        self._generate_res_save_path()

    def run(self):
        """
        Convenience function to run analysis steps.

        1. Load data or compute
        2. Create results directory if needed
        3. Load results or
        4. Compute results if needed/desired.
        """

        # Step 1: load data
        if self.subject_data is None:
            if self.ana_requires_data:
                self.load_data()

                # save data if it doesn't exist
                if not os.path.exists(self.save_file):
                    if self.auto_save_data:
                        self.save_data()

        # Step 2: create (if needed) directory to save/load results
        self._make_res_dir()

        # Step 3: if we want to load results instead of computing, try to load
        if self.load_res_if_file_exists:
            self.load_res_data()

        # Step 4: if res file doesn't exist
        if ((self.res_save_file is None) or not (os.path.exists(self.res_save_file))) or self.force_analysis:

            # Step 4A: run the subclass analysis
            if not self.do_not_compute_res:
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
            self.res = {**self.res, **joblib.load(self.res_save_file)}
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
        This should be overridden by an an analysis subclass. Should set .res_save_dir to a path to the directory
        where results should be saved.

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

        self._generate_res_save_path()
        self.res_save_file = os.path.join(self.res_save_dir, self.subject + '_' + self.res_str)
        if not os.path.exists(self.res_save_dir):
            try:
                os.makedirs(self.res_save_dir)
            except OSError:
                pass