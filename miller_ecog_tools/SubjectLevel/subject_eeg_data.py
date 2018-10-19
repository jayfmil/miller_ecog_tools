import os
import numpy as np

from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectEEGData(SubjectDataBase):
    """
    Subclass of SubjectDataBase for loading/saving spectral analyses of EEG/ECoG/LFP data.

    This class helps with computation of power values and allows for specification of the type of events you
    want to examine, the frequencies at which to compute power, the start and stop time of the power compuation
    relative to the events, and more. See below for list of attributes and their functions.

    # whether to load bipolar pairs of electrodes or monopolar contacts
    self.bipolar = True

    # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
    self.mono_avg_ref = False

    # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
    # that will be applied to the events. Function must return events dataframe.
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
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2:d}_freqs_{3:.3f}_{4:.3f}_{5}/{6}/{7}/{8}_bins/{9}/{10}/power'
    attrs_in_save_str = ['base_dir', 'task', 'freqs', 'event_type','start_time', 'end_time', 'time_bins', 'subject', 'montage']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectEEGData, self).__init__(task=task, subject=subject, montage=montage)

        # whether to load bipolar pairs of electrodes or monopolar contacts
        self.bipolar = True

        # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
        self.mono_avg_ref = False

        # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
        # that will be applied to the events. Function must return events dataframe.
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
        self.mean_over_time = True
        self.time_bins = None
        self.use_mirror_buf = False

        # this will hold the a dataframe of electrode locations/information after load_data() is called
        self.elec_info = None

    def load_data(self):
        """
        Call super's load data, and then additionally cast data to float32 to take up less space.
        """
        super(SubjectEEGData, self).load_data()
        if self.subject_data is not None:
            self.subject_data.data = self.subject_data.data.astype('float32')
            self.elec_info = RAM_helpers.load_elec_info(self.subject, self.montage, self.bipolar)

    def compute_data(self):
        """
        Does the power computation. Bulk of the work is handled by RAM_helpers.

        Sets .elec_info and returns subject_data
        """

        # load subject events
        events = RAM_helpers.load_subj_events(self.task, self.subject, self.montage, as_df=True, remove_no_eeg=True)

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
                                                 cluster_pool=self.pool,
                                                 log_power=self.log_power,
                                                 time_bins=self.time_bins,
                                                 noise_freq=self.noise_freq,
                                                 elec_scheme=self.elec_info,
                                                 resample_freq=self.resample_freq,
                                                 do_average_ref=self.mono_avg_ref,
                                                 mean_over_time=self.mean_over_time,
                                                 use_mirror_buf=self.use_mirror_buf,
                                                 loop_over_chans=True)
        return subject_data

    ##########################################################################################################
    # ECoG HELPERS - Some useful methods that we commonly perform for this type of data can go here. Now all #
    # subclasses will have access to this functionality                                                      #
    ##########################################################################################################
    def zscore_data(self):
        """
        Give all our subclasses easy access to zscoring the data.

        Returns a numpy array the same shape as the data.

        """
        return RAM_helpers.zscore_by_session(self.subject_data)

    def normalize_power_spectrum(self, event_dim_str='event'):
        """
        Normalized .subject_data power spectra so that the mean power spectrum is centered at zero was an SD of 1, as
        in Manning et al., 2009

        Returns a numpy array the same shape as the data.
        """

        sessions = self.subject_data[event_dim_str].data['session']
        norm_spectra = np.empty(self.subject_data.shape, dtype='float32')
        uniq_sessions = np.unique(sessions)
        for sess in uniq_sessions:
            sess_inds = sessions == sess

            m = np.mean(self.subject_data[sess_inds], axis=1)
            m = np.mean(m, axis=0)
            s = np.std(self.subject_data[sess_inds], axis=1)
            s = np.mean(s, axis=0)
            norm_spectra[sess_inds] = (self.subject_data[sess_inds] - m) / s

        return norm_spectra

    def bin_electrodes_by_region(self, elec_column1='stein.region', elec_column2='ind.region',
                                 x_coord_column='ind.x', roi_dict=None):
        """

        Given that we often want to look at effecfs based on brain region, this will take a subject's electrode info
        and bin it into broad ROIs based on lobe and hemisphere. In the project's terminology, `elec_column1` should
        usually be the 'loc_tag' information.

        Parameters
        ----------
        elec_column1: str
            DataFrame column to use for localization info.
        elec_column2: str
            Additional secondary DataFrame column to use.
        x_coord_column: str
            Column specifying the x-coordinate of each electrode. Used to determine left vs right hemisphere.
            Positive values are right hemisphere.
        roi_dict: dict
            A mapping of elec_column1/elec_column2 values to broader ROIs. If not given, the default will be used:

            {'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                                 'Right CA3', 'Right DG', 'Right Sub'],
             'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC'],
             'Frontal': ['parsopercularis', 'parsorbitalis', 'parstriangularis', 'caudalmiddlefrontal',
                                    'rostralmiddlefrontal', 'superiorfrontal'],
            'Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal'],
            'Parietal': ['inferiorparietal', 'supramarginal', 'superiorparietal', 'precuneus'],
            'Occipital' ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']}


        Returns
        -------
        A pandas.DataFrame with columns 'region' and 'hemisphere'.

        """
        if self.elec_info is None:
            print('{}: please load data before trying to bin electrode locations'.format(self.subject))
            return

        # smoosh the columns together, with the first column taking precedence
        regions = self.elec_info[elec_column1].fillna(self.elec_info[elec_column2]).fillna(value='')

        # if no dictionary is providing, use this
        if roi_dict is None:
            roi_dict = {'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                                 'Right CA3', 'Right DG', 'Right Sub'],
                        'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC'],
                        'Frontal': ['parsopercularis', 'parsorbitalis', 'parstriangularis', 'caudalmiddlefrontal',
                                    'rostralmiddlefrontal', 'superiorfrontal'],
                        'Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal'],
                        'Parietal': ['inferiorparietal', 'supramarginal', 'superiorparietal', 'precuneus'],
                        'Occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']}

        # get ROI for each electrode. THIS GETS THE FIRST, IF IT IS IN MULTIPLE SOMEHOW
        elec_region_list = [''] * len(regions)
        for e, elec_region in enumerate(regions):
            for roi in roi_dict.keys():
                if elec_region in roi_dict[roi]:
                    elec_region_list[e] = roi
                    continue

        # get hemisphere
        elec_hemi_list = np.array(['right'] * len(regions))
        elec_hemi_list[self.elec_info[x_coord_column] < 0] = 'left'

        # make new DF
        region_df = self.elec_info[['label']].copy()
        region_df['region'] = elec_region_list
        region_df['hemi'] = elec_hemi_list

        return region_df



    ###################################################################################
    # dynamically update the data save location of we change the following attributes #
    ###################################################################################
    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, x):
        self._base_dir = x
        self._update_save_path()

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
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, x):
        self._event_type = x
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
            event_type_str = self.event_type.__name__ if callable(self.event_type) else '_'.join(self.event_type)
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
                                                               event_type_str,
                                                               time_str,
                                                               num_tbins,
                                                               self.subject,
                                                               self.montage)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')
