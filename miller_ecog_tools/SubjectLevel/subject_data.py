import joblib
import os
import numpy as np
from miller_ecog_tools import RAM_helpers


class SubjectData(object):
    """
    Base class for handling data IO and computation. Override .compute_data() to handle your specific type of data.
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


class SubjectEEGData(SubjectData):
    """
    Subclass of SubjectData for loading/saving spectral analyses of EEG/ECoG/LFP data.

    Currently, this class helps with computation of power values and allows for specification of the type of events you
    want to examine, the frequencies at which to compute power, the start and stop time of the power compuation
    relative to the events, and more. See below for list of attributes and their functions.



    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2:d}_freqs_{3:.3f}_{4:.3f}_{5}/{6}/{7}_bins/{8}/{9}/power'
    attrs_in_save_str = ['base_dir', 'task', 'freqs', 'start_time', 'end_time', 'time_bins', 'subject', 'montage']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectEEGData, self).__init__(task=task, subject=subject, montage=montage)

        # whether to load bipolar pairs of electrodes or monopolar contacts
        self.bipolar = True

        # if doing monopolar, this will take the average reference of all electrodes
        self.mono_avg_ref = True # NOT YET IMPLEMENTED

        # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
        # that will be applied to the events. Function must return a filtered set of events
        self.event_type = ['WORD']

        # power computation settings
        self.start_time = -500
        self.end_time = 1500
        self.wave_num = 5
        self.buf_ms = 2000
        self.noise_freq = [58., 62.]
        self.resample_freq = None
        self.log_power = True
        self.freqs = np.logspace(np.log10(1), np.log10(200), 8)
        self.mean_over_time = False
        self.time_bins = None
        self.use_mirror_buf = False

        # this will hold the a dataframe of electrode locations/information after load_data() is called
        self.elec_info = None

    def load_data(self):
        """
        Call super's load data, and then additionally cast data to float32 to take up less space.
        """
        super(SubjectEEGData, self).load_data()
        if self.subject is None:
            self.subject_data.data = self.subject_data.data.astype('float32')

    def compute_data(self):
        """
        Does the power computation. Bulk of the work is handled by RAM_helpers.

        Sets .elec_info and returns subject_data
        """

        # load subject events
        events = RAM_helpers.load_subj_events(self.task, self.subject, self.montage, as_df=True)

        # load electrode info
        self.elec_info = RAM_helpers.load_elec_info(self.subject, self.montage, self.bipolar)

        # filter events if desired
        if callable(self.event_type):
            events_for_computation = self.event_type(events)
        else:
            event_type = [self.event_type] if isinstance(self.event_type, str) else self.event_type
            events_for_computation = events[events['type'].isin(event_type)]

        # compute power with RAM_helper function
        subject_data = RAM_helpers.compute_power(events_for_computation,
                                                 self.freqs,
                                                 self.wave_num,
                                                 self.start_time,
                                                 self.end_time,
                                                 buf_ms=self.buf_ms,
                                                 log_power=self.log_power,
                                                 noise_freq=self.noise_freq,
                                                 elec_scheme=self.elec_info,
                                                 cluster_pool=self.pool,
                                                 resample_freq=self.resample_freq,
                                                 mean_over_time=self.mean_over_time,
                                                 use_mirror_buf=self.use_mirror_buf,
                                                 loop_over_chans=True)
        return subject_data

    ##################
    ## ECoG HELPERS ##
    ##################
    def zscore_data(self):
        """
        Give all our subclasses easy access to zscoring the data.

        Returns a numpy array the same shape as the data.

        """
        return RAM_helpers.zscore_by_session(self.subject_data)

    def bin_electrodes_by_region(self):
        pass

    ###################################################################################
    # dynamically update the data save location of we change the following attributes #
    ###################################################################################
    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, x):
        self._task = x
        self._update_save_path()

    @property
    def subject(self):
        return self._subject

    @subject.setter
    def subject(self, x):
        self._subject = x
        self._update_save_path()

    @property
    def montage(self):
        return self._montage

    @montage.setter
    def montage(self, x):
        self._montage = x
        self._update_save_path()

    @property
    def bipolar(self):
        return self._bipolar

    @bipolar.setter
    def bipolar(self, x):
        self._bipolar = x
        self._update_save_path()

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, x):
        self._start_time = x
        self._update_save_path()

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, x):
        self._end_time = x
        self._update_save_path()

    @property
    def freqs(self):
        return self._freqs

    @freqs.setter
    def freqs(self, x):
        self._freqs = x
        self._update_save_path()

    @property
    def time_bins(self):
        return self._time_bins

    @time_bins.setter
    def time_bins(self, x):
        self._time_bins = x
        self._update_save_path()

    def _update_save_path(self):
        if np.all([hasattr(self, x) for x in SubjectEEGData.attrs_in_save_str]):
            num_tbins = '1' if self.time_bins is None else str(self.time_bins.shape[0])
            bipol_str = 'bipol' if self.bipolar else 'mono'
            f1 = self.freqs[0]
            f2 = self.freqs[-1]

            if callable(self.start_time):
                time_str = self.start_time.__name__ + '_' + self.end_time.__name__
            else:
                t1 = self.start_time[0] if isinstance(self.start_time, list) else self.start_time
                t2 = self.end_time[0] if isinstance(self.end_time, list) else self.end_time
                time_str = '{}_start_{}_stop'.format(t1, t2)

            # auto set save_dir and save_file and res_save_dir
            self.save_dir = SubjectEEGData.save_str_tmp.format(self.base_dir,
                                                               self.task,
                                                               len(self.freqs), f1, f2, bipol_str,
                                                               time_str,
                                                               num_tbins,
                                                               self.subject,
                                                               self.montage)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')
