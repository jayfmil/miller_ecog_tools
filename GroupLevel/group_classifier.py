import os
import pdb
import cluster_helper.cluster
import default_analyses
import logging
import numpy as np
from datetime import datetime
from SubjectLevel.subject import Subject


def setup_logger(fname):
    """

    """
    log_str = '/scratch/jfm2/python/%s_' %fname + datetime.now().strftime('%H_%M_%d_%m_%Y.log')
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=log_str)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.ERROR)


class GroupAnalysis(object):
    """

    """

    def __init__(self, analysis_name=None, open_pool=False, n_jobs=100, **kwargs):

        self.analysis_name = analysis_name
        self.open_pool = open_pool
        self.n_jobs = n_jobs

        # list that will hold all the Subjects objects
        self.subject_objs = None

        # kwargs will override defaults
        self.kwargs = kwargs

    def process(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """
        params = default_analyses.get_default_analysis_params(self.analysis_name)

        # if we have a parameters dictionary
        if not params:
            print 'Invalid analysis_name'
        else:

            # set up logger to log errors for this run
            setup_logger(self.analysis_name)

            # adjust default params
            for key in self.kwargs:
                params[key] = self.kwargs[key]

            # open a pool for parallel processing if desired
            if self.open_pool:
                with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", num_jobs=self.n_jobs,
                                                         cores_per_job=1,
                                                         extra_params={"resources": "h_vmem=12G"}) as pool:
                    params['pool'] = pool
                    subject_list = self.process_subjs(params)
            else:
                subject_list = self.process_subjs(params)
            self.subject_objs = subject_list

    @staticmethod
    def process_subjs(params):
        """

        """
        # will append Subject objects to this list
        subject_list = []

        for subj in params['subjs']:

            # Some subjects have some weird issues with their data or behavior that cause trouble, hence the try
            try:
                curr_subj = Subject(task=params['task'], subject=subj)

                for key in params:
                    if key != 'subjs':
                        setattr(curr_subj, key, params[key])

                # load the data to be classified
                curr_subj.load_data()

                # save data to disk
                if not os.path.exists(curr_subj.save_file):
                    curr_subj.save_data()

                # run the classifier
                curr_subj.run_classifier()
                print('%s: %.3f AUC.' % (curr_subj.subj, curr_subj.class_res['auc']))

                # don't need to store the original data in our results, so remove it
                curr_subj.subject_data = None
                if curr_subj.class_res is not None:
                    subject_list.append(curr_subj)

            # log the error and move on
            except Exception, e:
                print('ERROR PROCESSING %s.' % subj)
                logging.error('ERROR PROCESSING %s' % subj)
                logging.error(e, exc_info=True)

        return subject_list


    # should use pandas?