import os
import numpy as np

from miller_ecog_tools.Utils import ecog_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectRamEEGData(SubjectDataBase):
    """
    Subclass of SubjectDataBase for loading/saving spectral analyses of EEG/.

    # whether to load bipolar pairs of electrodes or monopolar contacts
    self.bipolar = True

    # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
    self.mono_avg_ref = False

    # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
    # that will be applied to the events. Function must return events dataframe.
    self.event_type = ['WORD']

    # eeg loading settings
    self.start_time = -500
    self.end_time = 1600
    self.buf_ms = 2000
    self.noise_freq = [58., 62.]
    self.resample_freq = None
    self.use_mirror_buf = False

    # this will hold the a dataframe of electrode locations/information after load_data() is called
    self.elec_info = None
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2}/{3}/{4}/{5}/{6}/eeg'
    attrs_in_save_str = ['base_dir', 'task', 'event_type', 'start_time', 'end_time', 'subject', 'montage', 'bipolar']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectRamEEGData, self).__init__(task=task, subject=subject, montage=montage)

        # whether to load bipolar pairs of electrodes or monopolar contacts
        self.bipolar = True

        # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
        self.mono_avg_ref = False

        # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
        # that will be applied to the events. Function must return events dataframe.
        self.event_type = ['WORD']

        # power computation settings
        self.start_time = -500
        self.end_time = 1600
        self.buf_ms = 2000
        self.noise_freq = [58., 62.]
        self.resample_freq = None
        self.demean_eeg = False
        self.use_mirror_buf = False

        # this will hold the a dataframe of electrode locations/information after load_data() is called
        self.elec_info = None

    def load_data(self):
        """
        Call super's load data, and then additionally cast data to float32 to take up less space.
        """
        super(SubjectRamEEGData, self).load_data()
        if self.subject_data is not None:
            self.subject_data.data = self.subject_data.data.astype('float32')
            self.elec_info = ecog_helpers.load_elec_info(self.subject, self.montage, self.bipolar)

    def compute_data(self):
        """
        Does the power computation. Bulk of the work is handled by RAM_helpers.

        Sets .elec_info and returns subject_data
        """

        # load subject events
        events = ecog_helpers.load_subj_events(self.task, self.subject, self.montage, as_df=True, remove_no_eeg=True)

        # load electrode info
        self.elec_info = ecog_helpers.load_elec_info(self.subject, self.montage, self.bipolar)

        # filter events if desired
        if callable(self.event_type):
            events = self.event_type(events)
        else:
            event_type = [self.event_type] if isinstance(self.event_type, str) else self.event_type
            events = events[events['type'].isin(event_type)]

        # load eeg
        eeg = ecog_helpers.load_eeg(events,
                                    self.start_time,
                                    self.end_time,
                                    buf_ms=self.buf_ms,
                                    demean=self.demean_eeg,
                                    elec_scheme=self.elec_info,
                                    noise_freq=self.noise_freq,
                                    resample_freq=self.resample_freq,
                                    do_average_ref=self.mono_avg_ref)

        return eeg

    ##########################################################################################################
    # ECoG HELPERS - Some useful methods that we commonly perform for this type of data can go here. Now all #
    # subclasses will have access to this functionality                                                      #
    ##########################################################################################################
    def zscore_data(self):
        """
        Give all our subclasses easy access to zscoring the data.

        Returns a numpy array the same shape as the data.

        """
        return ecog_helpers.zscore_by_session(self.subject_data)

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
        A pandas.DataFrame with columns 'region' and 'hemi'.

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
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, x):
        self._event_type = x
        self._update_save_path()

    def _update_save_path(self):
        if np.all([hasattr(self, x) for x in SubjectRamEEGData.attrs_in_save_str]):
            bipol_str = 'bipol' if self.bipolar else 'mono'
            event_type_str = self.event_type.__name__ if callable(self.event_type) else '_'.join(self.event_type)

            if callable(self.start_time):
                time_str = self.start_time.__name__ + '_' + self.end_time.__name__
            else:
                t1 = self.start_time[0] if isinstance(self.start_time, list) else self.start_time
                t2 = self.end_time[0] if isinstance(self.end_time, list) else self.end_time
                time_str = '{}_start_{}_stop'.format(t1, t2)

            # auto set save_dir and save_file and res_save_dir
            self.save_dir = SubjectRamEEGData.save_str_tmp.format(self.base_dir,
                                                                  self.task,
                                                                  event_type_str,
                                                                  time_str,
                                                                  bipol_str,
                                                                  self.subject,
                                                                  self.montage)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')