import logging
import cluster_helper.cluster
import numpy as np
from datetime import datetime

from miller_ecog_tools.SubjectLevel import subject
from miller_ecog_tools.SubjectLevel import Analyses


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


class Group(object):
    """
    Class to run a specified analyses on all subjects.
    """

    def __init__(self, analysis_name='SubjectSMEAnalysis', log_dir=None, open_pool=False, n_jobs=20,
                 subject_montage=None, task=None, **kwargs):

        # make sure we have a valid analyis
        if analysis_name not in Analyses.analysis_dict:
            print('Please enter a valid analysis name: \n{}'.format('\n'.join(list(Analyses.analysis_dict.keys()))))
            return

        self.analysis_name = analysis_name
        self.open_pool = open_pool
        self.n_jobs = n_jobs
        self.subject_montage = subject_montage
        self.task = task

        # list that will hold all the Subjects objects
        self.subject_objs = None

        # place to save the log
        self.log_dir = default_log_dir() if log_dir is None else log_dir

        # kwargs to set one the subject analysis objects
        self.kwargs = kwargs

    def process(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """

        # set up logger to log errors for this run
        setup_logger(self.analysis_name, self.log_dir)

        # open a pool for parallel processing if desired
        if self.open_pool:
            with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=self.n_jobs,
                                                     # cores_per_job=1, direct=False,
                                                     extra_params={"resources": "h_vmem=32G"}) as pool:

                subject_list = self.process_subjs(pool)
        else:
            subject_list = self.process_subjs()

        # save the list of subject results
        self.subject_objs = subject_list

    def process_subjs(self, pool=None):
        """
        Actually process the subjects here, and return a list of Subject objects with the results.
        """
        # will append Subject objects to this list
        subject_list = []

        for _, this_subj_montage in self.subject_montage.iterrows():
            this_subj_id = this_subj_montage.subject
            this_subj_montage = this_subj_montage.montage

            # create the subject analysis
            this_subj = subject.Subject(subject=this_subj_id, montage=this_subj_montage, task=self.task)
            this_subj.analysis_name = self.analysis_name

            # pass the pool along
            this_subj.analysis.pool = pool

            # set all the attributes
            for attr in self.kwargs.items():
                setattr(this_subj.analysis, attr[0], attr[1])

            # Some subjects have some weird issues with their data or behavior that cause trouble, hence the try
            try:

                # run the analysis
                print('Processing {} - {}'.format(this_subj_id, this_subj_montage))
                this_subj.analysis.run()

                # unload data and append to the list of subject objects
                this_subj.analysis.unload_data()
                subject_list.append(this_subj.analysis)

            # make sure to log any issues
            except Exception as e:
                print('ERROR PROCESSING %s.' % this_subj_id)
                logging.error('ERROR PROCESSING %s' % this_subj_id)
                logging.error(e, exc_info=True)

        return subject_list

