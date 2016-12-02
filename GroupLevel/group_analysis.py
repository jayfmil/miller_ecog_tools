import numpy as np
import cluster_helper.cluster
import default_analyses
from SubjectLevel.subject import Subject
import pdb

class GroupAnalysis(object):

    def __init__(self, analysis_name=None, open_pool=False, n_jobs=100):
        self.analysis_name = analysis_name
        self.open_pool = open_pool
        self.n_jobs = n_jobs

        # list that will hold all the Subjects objects
        self.subject_objs = None

    def process(self):
        """
        Opens a parallel pool or not, then hands off the work to process_subjs.
        """
        params = default_analyses.get_default_analysis_params(self.analysis_name)

        # if we have a parameters dictionary
        if not params:
            print 'Invalid analysis_name'
        else:

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
            curr_subj = Subject(task=params['task'], subject=subj)

            # set all the attributes based on the params
            for key in params:
                if key != 'subjs':
                    setattr(curr_subj, key, params[key])

            # load the data to be classified
            curr_subj.load_data()

            # run the classifier
            curr_subj.run_classifier()

            subject_list.append(curr_subj)
        return subject_list


    # should use pandas?