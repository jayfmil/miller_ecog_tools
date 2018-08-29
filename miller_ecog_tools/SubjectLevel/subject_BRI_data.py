import os
import numpy as np

from miller_ecog_tools.Utils import neurtex_bri_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectBRIData(SubjectDataBase):
    """
    Subclass of SubjectDataBase for computing spike-aligned eeg.
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2:.3f}_{3:.3f}_ms/{4}_ds/{5}/{6}/data'
    attrs_in_save_str = ['base_dir', 'task', 'start_spike_ms', 'stop_spike_ms',
                         'downsample_rate', 'spike_qual_to_use', 'subject']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRIData, self).__init__(task=task, subject=subject, montage=montage)

        # Spikes are either POTENTIAL or SPIKE. Default uses SPIKE only
        self.spike_qual_to_use = ['SPIKE']

        # start and stop relative relative to spike time for spike-triggered averages
        self.start_spike_ms = -500
        self.stop_spike_ms = 500

        # rate to downsample original ncs files
        self.downsample_rate = 1000

    # def load_data(self):
    #     """
    #     Call super's load data, and then additionally cast data to float32 to take up less space.
    #     """
    #     super(SubjectBRIData, self).load_data()
    #     if self.subject_data is not None:
    #         pass

    def compute_data(self):
        """
        Computes spike-aligned eeg data
        """

        # get list of channels
        file_dict = neurtex_bri_helpers.get_subj_files_by_sess(self.task, self.subject)

        # list to hold all channel data
        subject_data = []

        # loop over each session
        for session_id, session_dict in file_dict.items():

            # for each channel, load spike times of good clusters
            for channel_num in session_dict.keys():
                s_times, clust_num = neurtex_bri_helpers.load_spikes_cluster_with_qual(session_dict, channel_num,
                                                                                       quality=self.spike_qual_to_use)

                # if we have spikes for this channel, load spike-aligned eeg
                chan_eeg = neurtex_bri_helpers.load_eeg_from_spike_times(s_times, clust_num,
                                                                         session_dict[session_id][clust_num]['ncs'],
                                                                         self.start_spike_ms, self.stop_spike_ms,
                                                                         downsample_freq=self.downsample_rate)
                # cast to 32 bit for memory issues
                chan_eeg.data = chan_eeg.data.astype('float32')
                subject_data.append(chan_eeg)

        return subject_data


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
    def start_spike_ms(self):
        return self._start_spike_ms

    @start_spike_ms.setter
    def start_spike_ms(self, x):
        self._start_spike_ms = x
        self._update_save_path()

    @property
    def stop_spike_ms(self):
        return self._stop_spike_ms

    @stop_spike_ms.setter
    def stop_spike_ms(self, x):
        self._stop_spike_ms = x
        self._update_save_path()

    @property
    def downsample_rate(self):
        return self._downsample_rate

    @downsample_rate.setter
    def downsample_rate(self, x):
        self._downsample_rate = x
        self._update_save_path()

    @property
    def spike_qual_to_use(self):
        return self._spike_qual_to_use

    @spike_qual_to_use.setter
    def spike_qual_to_use(self, x):
        self._spike_qual_to_use = x
        self._update_save_path()

    def _update_save_path(self):
        if np.all([hasattr(self, x) for x in SubjectBRIData.attrs_in_save_str]):

            # auto set save_dir and save_file and res_save_dir
            self.save_dir = SubjectBRIData.save_str_tmp.format(self.base_dir,
                                                               self.task,
                                                               self.start_spike_ms,
                                                               self.stop_spike_ms,
                                                               self.spike_qual_to_use,
                                                               self.subject)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')
