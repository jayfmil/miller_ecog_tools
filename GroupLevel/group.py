import logging
import os
from datetime import datetime
import cluster_helper.cluster
import numpy as np
import default_analyses
import exclusions
import pdb


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


class Group(object):
    """
    Class to run a specified analyses on all subjects.
    """

    def __init__(self, analysis='classify_enc', subject_settings='default', open_pool=False, n_jobs=50,
                 base_dir='/scratch/jfm2/python', **kwargs):

        self.analysis = analysis
        self.subject_settings = subject_settings
        self.open_pool = open_pool
        self.n_jobs = n_jobs
        self.base_dir = base_dir

        # list that will hold all the Subjects objects
        self.subject_objs = None

        # kwargs will override defaults
        self.kwargs = kwargs

    def process(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """
        params = default_analyses.get_default_analysis_params(self.analysis, self.subject_settings)

        # if we have a parameters dictionary
        if not params:
            print('Invalid analysis or subject settings')
        else:

            # set up logger to log errors for this run
            setup_logger(self.analysis + '_' + self.subject_settings, self.base_dir)

            # adjust default params
            for key in self.kwargs:
                params[key] = self.kwargs[key]
            params['base_dir'] = self.base_dir

            # open a pool for parallel processing if desired. subject data creation is parallelized here. If data
            # already exists, then there is no point (yet. some analyses might parallel other stuff)
            if self.open_pool:
                with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=self.n_jobs,
                                                         # cores_per_job=1, direct=False,
                                                         extra_params={"resources": "h_vmem=20G"}) as pool:
                    params['pool'] = pool
                    subject_list = self.process_subjs(params)
            else:
                subject_list = self.process_subjs(params)

            # save the list of subject results
            self.subject_objs = subject_list

    @staticmethod
    def process_subjs(params):
        """
        Actually process the subjects here, and return a list of Subject objects with the results.
        """
        # will append Subject objects to this list
        subject_list = []

        for subj in params['subjs']:

            # Some subjects have some weird issues with their data or behavior that cause trouble, hence the try
            try:

                # create the analysis object for the specific analysis, subject, task
                curr_subj = params['ana_class'](task=params['task'], subject=subj)

                # set the analysis parameters
                for key in params:
                    if key != 'subjs':
                        setattr(curr_subj, key, params[key])

                # load the data to be processed
                curr_subj.load_data()

                # save data to disk. Why am I doing this every time? There was a reason..
                curr_subj.save_data()

                # check first session
                # curr_subj = exclusions.remove_first_session_if_worse(curr_subj)

                # remove sessions without enough data
                curr_subj = exclusions.remove_abridged_sessions(curr_subj)

                # make sure we have above chance performance
                # if curr_subj.subject_data is not None:
                #     curr_subj = exclusions.remove_subj_if_at_chance(curr_subj)

                if curr_subj.subject_data is not None:

                    # call the analyses class run method
                    curr_subj.run()

                    # Don't want to store the raw data in our subject_list because it can potentially eat up a lot of
                    # memory. First add the skewness of the subjects distance errors, as this seems to be a good
                    # behavioral predictor of classification performance. Perhaps should move this to subject_data or
                    # subject_classifier.
                    # mean_err = np.mean(curr_subj.subject_data.events.data['distErr'])
                    # med_err = np.median(curr_subj.subject_data.events.data['distErr'])
                    # curr_subj.skew = mean_err - med_err
                    curr_subj.subject_data = None
                    if curr_subj.res is not None:
                        subject_list.append(curr_subj)

            # log the error and move on
            except Exception as e:
                print('ERROR PROCESSING %s.' % subj)
                logging.error('ERROR PROCESSING %s' % subj)
                logging.error(e, exc_info=True)

        return subject_list

    def compute_pow_two_series(self):
        """
        This convoluted line computes a series powers of two up to and including one power higher than the
        frequencies used. Will use this as our axis ticks and labels so we can have nice round values.
        """
        return np.power(2, range(int(np.log2(2 ** (int(self.subject_objs[0].freqs[-1]) - 1).bit_length())) + 1))
