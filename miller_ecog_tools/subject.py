import os
import joblib


def create_subject(task, subject, montage=0, analysis_name=None):
    """Returns an object of the class specified in analysis_name. This is really just a helper function, you can always
    import the analysis class directly. Analyses live in miller_ecog_tools.SubjectLevel.Analyses

    Parameters
    ----------
    task: str
        The experiment name (ex: TH1, FR1, ...).
    subject: str
        The subject identifier code
    montage: int
        The montage number of the subject's electrodes (if not applicable, leave as 0)
    analysis_name: str
        The name of the analysis class you wish to instantiate. If not entered, a list of possible analyses will be
         printed.

    Returns
    -------
    Instantiated analysis class
    """
    from miller_ecog_tools.SubjectLevel import Analyses
    if analysis_name is None:
        print('You must enter one of the following as an analysis_name:\n')
        for this_ana in Analyses.analysis_dict.keys():
            print('{}\n{}'.format(this_ana, Analyses.analysis_dict[this_ana].__doc__))
    else:
        return Analyses.analysis_dict[analysis_name](task, subject, montage)


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
        raise NotImplementedError

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