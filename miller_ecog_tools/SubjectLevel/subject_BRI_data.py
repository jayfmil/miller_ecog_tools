import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


from miller_ecog_tools.Utils import neurtex_bri_helpers as bri_helpers
from miller_ecog_tools.subject import SubjectDataBase


class SubjectBRIData(SubjectDataBase):
    """
    Subclass of SubjectDataBase for computing spike-aligned eeg.
    """

    # Automatically set up the save directory path based on this design. See properties at the end of file. Any time
    # one of these attributes is modified, the save path will be automatically updated.
    save_str_tmp = '{0}/{1}/time_{2:d}_{3:d}_ms/{4}_noise/{5:d}_ds/{6}_qual/{7}/data'
    attrs_in_save_str = ['base_dir', 'task', 'start_spike_ms', 'stop_spike_ms', 'noise_freq',
                         'downsample_rate', 'spike_qual_to_use', 'subject']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRIData, self).__init__(task=task, subject=subject, montage=montage)

        # Spikes are either POTENTIAL or SPIKE. Default uses SPIKE only
        self.spike_qual_to_use = ['SPIKE']

        # start and stop relative relative to spike time for spike-triggered averages
        self.start_spike_ms = -1000
        self.stop_spike_ms = 1000

        # rate to downsample original ncs files
        self.downsample_rate = 1000

        # specify if we are computing power spectra
        self.do_compute_power = True
        self.ds_rate_pow = 250
        self.freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # line noise frequency to filter out
        self.noise_freq = [58., 62.]

    def compute_data(self):
        """
        Computes spike-aligned eeg data
        """

        # make sure the directory structure exists for saving file
        self._make_save_dir()

        # get list of channels
        file_dict = bri_helpers.get_subj_files_by_sess(self.task, self.subject)

        # will hold a list of hdf5 keys to store the event data. Can't append dataframes to hdf5 in the with statement,
        # so will do it after
        event_keys_dict = {}

        # list to hold all channel data
        with h5py.File(self.save_file, 'w') as subject_data:

            # loop over each session
            print('{}: Computing spike-aligned EEG for {} sessions.'.format(self.subject, len(file_dict)))
            for session_id, session_dict in file_dict.items():
                sess_grp = subject_data.create_group(session_id)

                # for each channel, load spike times of good clusters
                for channel_num in tqdm(session_dict.keys()):
                    s_times, clust_nums = bri_helpers.load_spikes_cluster_with_qual(session_dict, channel_num,
                                                                                    quality=self.spike_qual_to_use)

                    # if we have spikes for this channel, proceed
                    if s_times.size > 0:

                        # first create data frame
                        df = pd.DataFrame(data=np.stack([s_times, clust_nums], -1), columns=['stTime', 'cluster_num'])
                        df['session'] = session_id

                        # get some extra info: cluster region and hemisphere. Add to dataframe that will be stored in
                        # the event coord of chan_eeg
                        region, hemi = bri_helpers.get_localization_by_sess(self.subject, session_id, channel_num, clust_nums)
                        df['region'] = region
                        df['hemi'] = hemi

                        # add channel group to hdf5 file
                        chan_grp = sess_grp.create_group(str(channel_num))

                        # loop over each cluster in this channel
                        for this_cluster in df.cluster_num.unique():
                            df_clust = df[df.cluster_num == this_cluster]

                            # load spike-aligned eeg
                            clust_eeg = bri_helpers.load_eeg_from_spike_times(df_clust,
                                                                              session_dict[channel_num]['ncs'],
                                                                              self.start_spike_ms,
                                                                              self.stop_spike_ms,
                                                                              noise_freq=self.noise_freq,
                                                                              downsample_freq=self.downsample_rate)

                            # cast to 32 bit for memory issues
                            clust_eeg.data = clust_eeg.data.astype('float32')

                            # create cluster group within this channel group
                            clust_grp = chan_grp.create_group(str(this_cluster))

                            # add our data for this channel and cluster to the hdf5 file
                            clust_grp.create_dataset('ST_eeg', data=clust_eeg.data)
                            clust_grp.attrs['time'] = clust_eeg.time.data
                            clust_grp.attrs['channel'] = str(clust_eeg.channel.data[0])
                            clust_grp.attrs['samplerate'] = float(clust_eeg.samplerate.data)

                            # store path to where we will append the event data
                            this_key = clust_grp.name + '/event'
                            event_keys_dict[this_key] = pd.DataFrame.from_records(clust_eeg.event.data)

                        # also, compute power spectra for channel. Sorry, this reloads the channel data and is
                        # inefficient
                        if self.do_compute_power:
                            power_spectra = bri_helpers.power_spectra_from_spike_times(s_times, clust_nums,
                                                                                       session_dict[channel_num]['ncs'],
                                                                                       self.start_spike_ms,
                                                                                       self.stop_spike_ms,
                                                                                       self.freqs,
                                                                                       noise_freq=self.noise_freq,
                                                                                       downsample_freq=self.ds_rate_pow,
                                                                                       mean_over_spikes=True)

                            # store the power spectra for each cluster seperately in the hdf5 file
                            for cluster_key in power_spectra.keys():
                                clust_grp = chan_grp[str(cluster_key)]
                                clust_grp.create_dataset('power_spectra', data=power_spectra[cluster_key])

        # append all events from all channels to file
        for event_key in event_keys_dict.keys():
            event_keys_dict[event_key].to_hdf(self.save_file, event_key, mode='a')

        return h5py.File(self.save_file, 'r')

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

                if self.load_data_if_file_exists:
                    print('%s: subject_data already exists, loading.' % self.subject)
                    self.subject_data = h5py.File(self.save_file, 'r')
                else:
                    print('%s: subject_data exists, but redoing anyway.' % self.subject)

            else:
                print('%s: subject_data exists, but redoing anyway.' % self.subject)
                return

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
        self.subject_data.close()

    def save_data(self):
        """
        Data is saved on compute now that I switched this class to use hdf5 files. This does nothing.
        """
        pass

    def _make_save_dir(self):

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
    def noise_freq(self):
        return self._noise_freq

    @noise_freq.setter
    def noise_freq(self, x):
        self._noise_freq = x
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
            noise_str = '_'.join([str(x) for x in self.noise_freq]) if self.noise_freq else 'no_filt'
            self.save_dir = SubjectBRIData.save_str_tmp.format(self.base_dir,
                                                               self.task,
                                                               self.start_spike_ms,
                                                               self.stop_spike_ms,
                                                               noise_str,
                                                               self.downsample_rate,
                                                               '_'.join(self.spike_qual_to_use),
                                                               self.subject)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.hdf5')
