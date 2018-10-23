import os
import numpy as np

from miller_ecog_tools.Utils import RAM_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectEventsRAMData(SubjectDataBase):
    """
    Subclass of SubjectDataBase where the data is behavioral events only. The class can be useful when doing analyses
    where saving out intermediate data like power values is not appropriate. Here, .subject_data will be the events
    returned by RAM_helpers and filtered by .event_type

    # whether to load bipolar pairs of electrodes or monopolar contacts
    self.bipolar = True

    # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
    self.mono_avg_ref = False

    # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
    # that will be applied to the events. Function must return events dataframe.
    self.event_type = ['WORD']

    # this will hold the a dataframe of electrode locations/information after load_data() is called
    self.elec_info = None
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2}/{3}/{4}/data'
    attrs_in_save_str = ['base_dir', 'task', 'event_type', 'subject', 'montage']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectEventsRAMData, self).__init__(task=task, subject=subject, montage=montage)

        # whether to load bipolar pairs of electrodes or monopolar contacts
        self.bipolar = True

        # This will load eeg and compute the average reference before computing power. Recommended if bipolar = False.
        self.mono_avg_ref = False

        # the event `type` to filter the events to. This can be a string, a list of strings, or it can be a function
        # that will be applied to the events. Function must return events dataframe.
        self.event_type = ['WORD']

        # this will hold the a dataframe of electrode locations/information after load_data() is called
        self.elec_info = None

    def load_data(self):
        """
        Call super's load data. Here, loads events and filters to specific type
        """
        super(SubjectEventsRAMData, self).load_data()

        # also load electrode info
        self.elec_info = RAM_helpers.load_elec_info(self.subject, self.montage, self.bipolar)

    def compute_data(self):
        """
        Does the power computation. Just loads events and electrode info.

        Sets .elec_info and returns subject_data
        """

        # load subject events
        events = RAM_helpers.load_subj_events(self.task, self.subject, self.montage, as_df=True, remove_no_eeg=True)

        # filter events if desired
        if callable(self.event_type):
            events = self.event_type(events)
        else:
            event_type = [self.event_type] if isinstance(self.event_type, str) else self.event_type
            events = events[events['type'].isin(event_type)]

        return events

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
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, x):
        self._event_type = x
        self._update_save_path()

    def _update_save_path(self):
        if np.all([hasattr(self, x) for x in SubjectEventsRAMData.attrs_in_save_str]):
            event_type_str = self.event_type.__name__ if callable(self.event_type) else '_'.join(self.event_type)

            # auto set save_dir and save_file and res_save_dir
            self.save_dir = SubjectEventsRAMData.save_str_tmp.format(self.base_dir,
                                                                     self.task,
                                                                     event_type_str,
                                                                     self.subject,
                                                                     self.montage)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')
