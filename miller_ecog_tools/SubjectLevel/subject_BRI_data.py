import os
import numpy as np
from tqdm import tqdm


from miller_ecog_tools.Utils import neurtex_bri_helpers as bri_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectBRIData(SubjectDataBase):
    """
    Subclass of SubjectDataBase for computing spike-aligned eeg.
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/{2:d}_{3:d}_ms/{4:d}_ds/{5}_qual/{6}/data'
    attrs_in_save_str = ['base_dir', 'task', 'start_spike_ms', 'stop_spike_ms',
                         'downsample_rate', 'spike_qual_to_use', 'subject']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRIData, self).__init__(task=task, subject=subject, montage=montage)

        # Spikes are either POTENTIAL or SPIKE. Default uses SPIKE only
        self.spike_qual_to_use = ['SPIKE']

        # start and stop relative relative to spike time for spike-triggered averages
        self.start_spike_ms = -500
        self.stop_spike_ms = 500
        self.buffer_ms = 1000

        # rate to downsample original ncs files
        self.downsample_rate = 1000

        # specify if we are computing power spectra
        self.do_compute_power = True
        self.ds_rate_pow = 250
        self.freqs = np.logspace(np.log10(1), np.log10(100),50)

    def compute_data(self):
        """
        Computes spike-aligned eeg data
        """

        # get list of channels
        file_dict = bri_helpers.get_subj_files_by_sess(self.task, self.subject)

        # list to hold all channel data
        subject_data = {}

        # loop over each session
        print('{}: Computing spike-aligned EEG for {} sessions.'.format(self.subject, len(file_dict)))
        for session_id, session_dict in file_dict.items():
            subject_data[session_id] = {}

            # for each channel, load spike times of good clusters
            for channel_num in tqdm(session_dict.keys()):
                s_times, clust_nums = bri_helpers.load_spikes_cluster_with_qual(session_dict, channel_num,
                                                                                quality=self.spike_qual_to_use)

                # if we have spikes for this channel, load spike-aligned eeg
                if s_times.size > 0:
                    chan_eeg = bri_helpers.load_eeg_from_spike_times(s_times, clust_nums,
                                                                     session_dict[channel_num]['ncs'],
                                                                     self.start_spike_ms, self.stop_spike_ms,
                                                                     buf_ms=self.buffer_ms,
                                                                     downsample_freq=self.downsample_rate)
                    # cast to 32 bit for memory issues
                    chan_eeg.data = chan_eeg.data.astype('float32')
                    subject_data[session_id][channel_num] = {}
                    subject_data[session_id][channel_num]['ST_eeg'] = chan_eeg

                    # also, compute power spectra for channel. Sorry, this reloads the channel data and is inefficient
                    if self.do_compute_power:
                        power_spectra = bri_helpers.power_spectra_from_spike_times(s_times, clust_nums,
                                                                                   session_dict[channel_num]['ncs'],
                                                                                   self.start_spike_ms,
                                                                                   self.stop_spike_ms,
                                                                                   self.freqs,
                                                                                   downsample_freq=self.ds_rate_pow,
                                                                                   mean_over_spikes=True)
                        subject_data[session_id][channel_num]['power_spectra'] = power_spectra

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
                                                               self.downsample_rate,
                                                               '_'.join(self.spike_qual_to_use),
                                                               self.subject)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.p')
