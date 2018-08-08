import os
import joblib


class Subject(object):
    """
    Base class upon which data and Analyses are built. A Subject has an associated task, subject code, and montage.

    In order to actually DO anything, you need to set the analyses_name attribute. See list of possible analyses with
    .list_possible_analyses().

    """

    def __init__(self, task=None, subject=None, montage=0):

        # set the attributes
        self.task = task
        self.subject = subject
        self.montage = montage

        # the setter will make self.analysis the analysis class. Now you don't have to import the specific
        # analysis module directly.
        self.analysis = None
        self.analysis_name = None

    # returns an initialized class based on the analysis name
    def _construct_analysis(self, analysis_name):
        from miller_ecog_tools.SubjectLevel import Analyses
        return Analyses.analysis_dict[analysis_name](self.task, self.subject, self.montage)

    @staticmethod
    def list_possible_analyses():
        from miller_ecog_tools.SubjectLevel import Analyses
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


class SubjectData(object):
    """
    Base class for handling data IO and computation. Override .compute_data() to handle your specific type of data.

    Methods:
        load_data()
        unload_data()
        save_data()
        compute_data()
    """

    def __init__(self, task=None, subject=None, montage=0):

        # attributes for identification of subject and experiment
        self.task = task
        self.subject = subject
        self.montage = montage

        # base directory to save data
        self.base_dir = self._default_base_dir()
        self.save_dir = None
        self.save_file = None

        # this will hold the subject data after load_data() is called
        self.subject_data = None

        # a parallel pool
        self.pool = None

        # settings for whether to load existing data
        self.load_data_if_file_exists = True  # this will load data from disk if it exists, instead of copmputing
        self.do_not_compute = False  # Overrules force_recompute. If this is True, data WILL NOT BE computed
        self.force_recompute = False  # Overrules load_data_if_file_exists, even if data exists

    def load_data(self):
        """
        Can load data if it exists, or can compute data.

        This sets .subject_data after loading
        """
        if self.subject is None:
            print('Attributes subject and task must be set before loading data.')
            return

        # if data already exist
        if os.path.exists(self.save_file):

            # load if not recomputing
            if not self.force_recompute:
                print('%s: Input data already exists, loading.' % self.subject)
                self.subject_data = joblib.load(self.save_file)

        # if do not exist
        else:

            # if not computing, don't do anything
            if self.do_not_compute:
                print('%s: subject_data does not exist, but not computing.' % self.subject)
                return

        # otherwise compute
        if self.subject_data is None:
            self.subject_data = self.compute_data()

    def unload_data(self):
        self.subject_data = None

    def save_data(self):
        """
        Saves self.data as a pickle to location defined by _generate_save_path.
        """
        if self.subject_data is None:
            print('Data must be loaded before saving. Use .load_data()')
            return

        if self.save_file is None:
            print('.save_file and .save_dir must be set before saving data.')

        # make directories if missing
        if not os.path.exists(os.path.split(self.save_dir)[0]):
            try:
                os.makedirs(os.path.split(self.save_dir)[0])
            except OSError:
                pass
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError:
                pass

        # pickle file
        joblib.dump(self.subject_data, self.save_file)

    def compute_data(self):
        """
        Override this. Should return data of some kind!

        """
        pass

    @staticmethod
    def _default_base_dir():
        """
        Set default save location based on OS. This gets set to default when you create the class, but you can set it
        to whatever you want later.
        """
        import platform
        import getpass
        uid = getpass.getuser()
        plat = platform.platform()
        if 'Linux' in plat:
            # assuming rhino
            base_dir = '/scratch/' + uid + '/python'
        elif 'Darwin' in plat:
            base_dir = '/Users/' + uid + '/python'
        else:
            base_dir = os.getcwd()
        return base_dir