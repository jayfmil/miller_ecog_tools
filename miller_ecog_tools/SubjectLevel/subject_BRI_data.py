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
    save_str_tmp = '{0}/{1}/{2}/time_{3:d}_{4:d}_ms/{5}_noise/{6:d}_ds/{7:d}_resamp/{8}_qual/{9}/data'
    attrs_in_save_str = ['base_dir', 'task', 'start_ms', 'stop_ms', 'noise_freq',
                         'downsample_rate', 'resample_rate', 'spike_qual_to_use', 'subject', 'do_event_locked']

    def __init__(self, task=None, subject=None, montage=0):
        super(SubjectBRIData, self).__init__(task=task, subject=subject, montage=montage)

        # Spikes are either POTENTIAL or SPIKE. Default uses SPIKE only
        self.spike_qual_to_use = ['SPIKE']

        # start and stop relative relative to spike time for spike-triggered averages
        self.start_ms = -1000
        self.stop_ms = 1000

        # rate to downsample original ncs files
        self.downsample_rate = 1000

        # rate to resample after downsampling
        self.resample_rate = None

        # specify if we are computing power spectra
        self.do_compute_power = True
        self.ds_rate_pow = 250
        self.freqs = np.logspace(np.log10(1), np.log10(100), 50)

        # line noise frequency to filter out
        self.noise_freq = [58., 62.]

        # specify if we want to load eeg event-locked or spike-locked
        self.do_event_locked = False

    def compute_data(self):
        """
        Computes spike-aligned or event-aligned eeg data for BRI datasets
        """

        # make sure the directory structure exists for saving file
        self._make_save_dir()

        # get list of channels
        file_dict = bri_helpers.get_subj_files_by_sess(self.task, self.subject)

        # will hold a list of hdf5 keys to store the event data. Can't append dataframes to hdf5 in the with statement,
        # so will do it after
        event_keys_dict = {}

        # if we are doing event-locked, load the subject's behavioral events
        if self.do_event_locked:
            beh_events = bri_helpers.load_subj_events(self.task, self.subject)

        # list to hold all channel data
        with h5py.File(self.save_file, 'w') as subject_data:
            for session_id, session_dict in file_dict.items():
                sess_grp = subject_data.create_group(session_id)

                # if doing event-locked, filter behavioral events to just this session
                if self.do_event_locked:
                    sess_beh_events = beh_events[beh_events.expID == session_id].reset_index(drop=True)

                # for each channel, load spike times of good clusters
                for channel_num in tqdm(session_dict.keys()):
                    s_times, clust_nums = bri_helpers.load_spikes_cluster_with_qual(session_dict, channel_num,
                                                                                    quality=self.spike_qual_to_use)

                    # get some extra info: cluster region and hemisphere. Add to dataframe that will be stored in
                    # the event coord of chan_eeg
                    region, hemi = bri_helpers.get_localization_by_sess(self.subject, session_id, channel_num,
                                                                        clust_nums)

                    # if we have spikes for this channel, proceed
                    if s_times.size > 0:

                        # either load eeg locked to events or locked to spikes
                        if not self.do_event_locked:
                            print('{}: Computing spike-aligned EEG for {} sessions.'.format(self.subject, len(file_dict)))
                            event_keys_dict = self._compute_spike_aligned(s_times, clust_nums, session_id, session_dict,
                                                                          sess_grp, channel_num, event_keys_dict,
                                                                          region, hemi)
                        else:
                            event_keys_dict = self._compute_event_aligned(s_times, clust_nums, sess_beh_events,
                                                                          session_dict, sess_grp, channel_num,
                                                                          event_keys_dict, region, hemi)

        # append all events from all channels to file
        for event_key in event_keys_dict.keys():
            event_keys_dict[event_key].to_hdf(self.save_file, event_key, mode='a')

        return h5py.File(self.save_file, 'r')

    def _compute_event_aligned(self, s_times, clust_nums, df, session_dict, sess_grp, channel_num,
                               event_keys_dict, region, hemi):
        """
        Method for computing the event-aligned data for a channel
        """

        # add region and hemi info to events
        df = df.assign(region=region[0])
        df = df.assign(hemi=hemi[0])

        # add channel group to hdf5 file
        chan_grp = sess_grp.create_group(str(channel_num))

        # load event-aligned eeg for this channel
        channel_eeg = bri_helpers.load_eeg_from_times(df, session_dict[channel_num]['ncs'],
                                                      self.start_ms,
                                                      self.stop_ms,
                                                      noise_freq=self.noise_freq,
                                                      downsample_freq=self.downsample_rate,
                                                      resample_freq=self.resample_rate)

        # cast to 32 bit for memory issues
        channel_eeg.data = channel_eeg.data.astype('float32')

        # add data to channel group
        chan_grp.create_dataset('ev_eeg', data=channel_eeg.data)
        chan_grp.attrs['time'] = channel_eeg.time.data
        chan_grp.attrs['channel'] = str(channel_eeg.channel.data[0])
        chan_grp.attrs['samplerate'] = float(channel_eeg.samplerate.data)

        # also store timestamps of spikes
        clust_grp_base = chan_grp.create_group('spike_times')
        for this_cluster in np.unique(clust_nums):
            clust_grp = clust_grp_base.create_group('cluster_'+str(this_cluster))
            this_cluster_times = s_times[clust_nums == this_cluster]

            # for each event, select spike times that fall within our timing winding
            for index, e in df.iterrows():

                # select the events, careful to convert ms to to microseconds
                inds = (this_cluster_times > e.stTime + self.start_ms*1000) & (this_cluster_times < e.start_ms + self.stop_ms*1000)

                # since hdf5 can't store variable length arrays in a single dataset, I'm saving the spike times for each
                # event as a seperate dataset. Kind of meh but it works
                clust_grp.create_dataset(str(index), data=np.array(this_cluster_times[inds]))

        # store path to where we will append the event data
        this_key = chan_grp.name + '/event'
        event_keys_dict[this_key] = pd.DataFrame.from_records(channel_eeg.event.data)

        return event_keys_dict

    def _compute_spike_aligned(self, s_times, clust_nums, session_id, session_dict, sess_grp, channel_num,
                               event_keys_dict, region, hemi):
        """
        Method for computing the spike-aligned data for a channel
        """

        # first create data frame
        df = pd.DataFrame(data=np.stack([s_times, clust_nums], -1), columns=['stTime', 'cluster_num'])
        df['session'] = session_id
        df['region'] = region
        df['hemi'] = hemi

        # add channel group to hdf5 file
        chan_grp = sess_grp.create_group(str(channel_num))

        # loop over each cluster in this channel
        for this_cluster in df.cluster_num.unique():
            df_clust = df[df.cluster_num == this_cluster]

            # load spike-aligned eeg
            clust_eeg = bri_helpers.load_eeg_from_times(df_clust,
                                                        session_dict[channel_num]['ncs'],
                                                        self.start_ms,
                                                        self.stop_ms,
                                                        noise_freq=self.noise_freq,
                                                        downsample_freq=self.downsample_rate,
                                                        resample_freq=self.resample_rate)

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
                                                                       self.start_ms,
                                                                       self.stop_ms,
                                                                       self.freqs,
                                                                       noise_freq=self.noise_freq,
                                                                       downsample_freq=self.ds_rate_pow,
                                                                       mean_over_spikes=True)

            # store the power spectra for each cluster seperately in the hdf5 file
            for cluster_key in power_spectra.keys():
                clust_grp = chan_grp[str(cluster_key)]
                clust_grp.create_dataset('power_spectra', data=power_spectra[cluster_key])

        return event_keys_dict

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
        self.subject_data = None

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
    def do_event_locked(self):
        return self._do_event_locked

    @do_event_locked.setter
    def do_event_locked(self, x):
        self._do_event_locked = x
        self._update_save_path()

    @property
    def subject(self):
        return self._subject

    @subject.setter
    def subject(self, x):
        self._subject = x
        self._update_save_path()

    @property
    def start_ms(self):
        return self._start_spike_ms

    @start_ms.setter
    def start_ms(self, x):
        self._start_spike_ms = x
        self._update_save_path()

    @property
    def stop_ms(self):
        return self._stop_spike_ms

    @stop_ms.setter
    def stop_ms(self, x):
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
    def resample_rate(self):
        return self._resample_rate

    @resample_rate.setter
    def resample_rate(self, x):
        self._resample_rate = x
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
                                                               'event_locked' if self.do_event_locked else 'spike_locked',
                                                               self.start_ms,
                                                               self.stop_ms,
                                                               noise_str,
                                                               self.downsample_rate,
                                                               self.resample_rate,
                                                               '_'.join(self.spike_qual_to_use),
                                                               self.subject)
            self.save_file = os.path.join(self.save_dir, self.subject + '_data.hdf5')
