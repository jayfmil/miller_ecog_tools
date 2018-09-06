import logging
import cluster_helper.cluster
from datetime import datetime

from miller_ecog_tools import subject
from miller_ecog_tools.GroupLevel import Analyses as GroupAnalyses
from miller_ecog_tools.SubjectLevel import Analyses as SubjectAnalyses


def setup_logger(fname, basedir):
    """
    This creates the logger to write all error messages when processing subjects.
    """
    log_str = '%s/%s' % (basedir, fname + '_' + datetime.now().strftime('%H_%M_%d_%m_%Y.log'))
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=log_str)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.ERROR)


def default_log_dir():
    """
    Set default save location based on OS. This gets set to default when you create the class, but you can set it
    to whatever you want later.
    """
    import platform
    import getpass
    import os
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


class GroupAnalysisPipeline(object):
    """
    Class to run multiple analyses on all subjects.
    """

    def __init__(self, analysis_name_list=[], analysis_params_list=[], log_dir=None, open_pool=False,
                 n_jobs=20, G_per_job=12, subject_montage=None, task=None):
        """

        Parameters
        ----------
        analysis_name_list: list of strings
            List of analysis names to run. They should be the name of a SubjectLevel analysis class.
        analysis_params_list: list of dictionaries
            List of dictionaries of attributes to set for each analysis
        log_dir: str
            Where to write the log file. If not given, will save to default location. See default_log_dir()
        open_pool: bool
            Whether to open a parallel pool for within subject computations.
        n_jobs: int
            If open_pool, this is how many jobs to create
        G_per_job: int
            If open_pool, how much memory to allocate for each job (in GB).
        subject_montage: pandas.DataFrame
            A dataframe with a row for each subject to iterate over. Must have a column 'subject'. Can also have a
            column 'montage'. If montage is not present, will use montage=0.
        task: str
            The experiment name

        Notes
        -----
        Use the .run() method to iterate over each subject.

        After .run() is complete:
            The .subject_objs attribute will hold a list of all the .res structures from the processed subjects.

            If there is a corresponding GroupLevel.Analysis class for the SubjectLevel.Analysis class, the
            .group_helpers attribute will be an instantiated version of that class. These classes accept the
            list of subject_objs as an input, and can be useful for statistics and group plotting. Corresponding
            GroupLevel classes should have the same name as the SubjectLevel analyses, just replace Subject in the class
            name with Group.

        """

        if (analysis_name_list is None) or (analysis_params_list is None):
            print('Both analysis_name_list and analysis_params_list must be entered.')
            return

        if len(analysis_name_list) != len(analysis_params_list):
            print('Both analysis_name_list and analysis_params_list must be the same length.')
            return

        self.analysis_name_list = analysis_name_list
        self.analysis_params_list = analysis_params_list
        self.open_pool = open_pool
        self.n_jobs = n_jobs
        self.G = G_per_job
        self.subject_montage = subject_montage
        self.task = task

        # place to save the log
        self.log_dir = default_log_dir() if log_dir is None else log_dir

        # list that will hold all the Subjects objects
        self.subject_objs = None

    def run(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """

        # set up logger to log errors for this run
        setup_logger('_'.join(self.analysis_name_list), self.log_dir)

        # open a pool for parallel processing if desired
        if self.open_pool:
            with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=self.n_jobs,
                                                     # cores_per_job=1, direct=False,
                                                     extra_params={"resources": "h_vmem={}G".format(self.G)}) as pool:

                subject_list = self.process_subj_pipeline(pool)
        else:
            subject_list = self.process_subj_pipeline()

        # save the list of subject results
        self.subject_objs = subject_list

    def process_subj_pipeline(self, pool=None):
        """
        Actually process the subjects here, and return a list of SubjectAnalysisPipeline objects.

        """
        # will append SubjectAnalysisPipeline objects to this list
        subject_list = []

        for _, this_subj_montage in self.subject_montage.iterrows():
            this_subj_id = this_subj_montage.subject
            this_subj_montage = this_subj_montage.montage if 'montage' in this_subj_montage else 0

            # create the subject analysis
            ana_dicts = self.analysis_params_list
            for this_dict in ana_dicts:
                this_dict['pool'] = pool
            this_subj = subject.SubjectAnalysisPipeline(self.task, this_subj_id, this_subj_montage,
                                                        self.analysis_name_list,
                                                        ana_dicts)

            # Some subjects have some weird issues with their data or behavior that cause trouble, hence the try
            try:

                # run the analysis
                print('Processing {} - {}'.format(this_subj_id, this_subj_montage))
                this_subj.run()

                # unload data and append to the list of subject objects
                for this_ana in this_subj.analyses:
                    this_ana.unload_data()
                subject_list.append(this_subj)

            # make sure to log any issues
            except Exception as e:
                print('ERROR PROCESSING %s.' % this_subj_id)
                logging.error('ERROR PROCESSING %s' % this_subj_id)
                logging.error(e, exc_info=True)

        return subject_list


class Group(object):
    """
    Class to run a specified analyses on all subjects.
    """

    def __init__(self, analysis_name='', log_dir=None, open_pool=False,
                 n_jobs=20, G_per_job=12, subject_montage=None, task=None, **kwargs):
        """

        Parameters
        ----------
        analysis_name: str
            The name of analysis to run. It should be the name of a SubjectLevel analysis class.
        log_dir: str
            Where to write the log file. If not given, will save to default location. See default_log_dir()
        open_pool: bool
            Whether to open a parallel pool for within subject computations.
        n_jobs: int
            If open_pool, this is how many jobs to create
        G_per_job: int
            If open_pool, how much memory to allocate for each job (in GB).
        subject_montage: pandas.DataFrame
            A dataframe with a row for each subject to iterate over. Must have a column 'subject'. Can also have a
            column 'montage'. If montage is not present, will use montage=0.
        task: str
            The experiment name
        kwargs
            Any additional keyword arguments will be set as attributes of the Analysis class.

        Notes
        -----
        Use the .run() method to iterate over each subject.

        After .run() is complete:
            The .subject_objs attribute will hold a list of all the .res structures from the processed subjects.

            If there is a corresponding GroupLevel.Analysis class for the SubjectLevel.Analysis class, the
            .group_helpers attribute will be an instantiated version of that class. These classes accept the
            list of subject_objs as an input, and can be useful for statistics and group plotting. Corresponding
            GroupLevel classes should have the same name as the SubjectLevel analyses, just replace Subject in the class
            name with Group.

        """

        # make sure we have a valid analyis
        if analysis_name not in SubjectAnalyses.analysis_dict:
            print('Please enter a valid analysis name: \n{}'.format('\n'.join(list(SubjectAnalyses.analysis_dict.keys()))))
            return

        self.analysis_name = analysis_name
        self.open_pool = open_pool
        self.n_jobs = n_jobs
        self.G = G_per_job
        self.subject_montage = subject_montage
        self.task = task

        # list that will hold all the Subjects objects
        self.subject_objs = None

        # place to save the log
        self.log_dir = default_log_dir() if log_dir is None else log_dir

        # kwargs to set one the subject analysis objects
        self.kwargs = kwargs

        # this will be used to give easy access to group analsyis specific function, such as custom plotting.
        # this will automatically add the
        self.group_helpers = None

    def run(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """

        # set up logger to log errors for this run
        setup_logger(self.analysis_name, self.log_dir)

        # open a pool for parallel processing if desired
        if self.open_pool:
            with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=self.n_jobs,
                                                     # cores_per_job=1, direct=False,
                                                     extra_params={"resources": "h_vmem={}G".format(self.G)}) as pool:

                subject_list = self.process_subjs(pool)
        else:
            subject_list = self.process_subjs()

        # save the list of subject results
        self.subject_objs = subject_list

        # add the group class, if exists
        if self.analysis_name.replace('Subject', 'Group') in GroupAnalyses.analysis_dict:
            if subject_list:
                ana_key = self.analysis_name.replace('Subject', 'Group')
                print('Setting .group_helpers to {} class.'.format(ana_key))
                self.group_helpers = GroupAnalyses.analysis_dict[ana_key](subject_list)

    def process_subjs(self, pool=None):
        """
        Actually process the subjects here, and return a list of Subject objects with the results.
        """
        # will append Subject objects to this list
        subject_list = []

        for _, this_subj_montage in self.subject_montage.iterrows():
            this_subj_id = this_subj_montage.subject
            this_subj_montage = this_subj_montage.montage if 'montage' in this_subj_montage else 0

            # create the subject analysis
            this_subj = subject.create_subject(self.task, this_subj_id, this_subj_montage,
                                               analysis_name=self.analysis_name)

            # pass the pool along
            this_subj.pool = pool

            # set all the attributes
            for attr in self.kwargs.items():
                setattr(this_subj, attr[0], attr[1])

            # Some subjects have some weird issues with their data or behavior that cause trouble, hence the try
            try:

                # run the analysis
                print('Processing {} - {}'.format(this_subj_id, this_subj_montage))
                this_subj.run()

                # unload data and append to the list of subject objects
                this_subj.unload_data()
                subject_list.append(this_subj)

            # make sure to log any issues
            except Exception as e:
                print('ERROR PROCESSING %s.' % this_subj_id)
                logging.error('ERROR PROCESSING %s' % this_subj_id)
                logging.error(e, exc_info=True)

        return subject_list

