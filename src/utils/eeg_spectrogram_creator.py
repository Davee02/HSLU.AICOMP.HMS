import librosa
import numpy as np
import pywt

from .utils import fill_nan_with_mean


class EEGSpectrogramGenerator:
    """
    A class for generating different types of spectrograms from EEG data.
    """

    def __init__(self, spectrogram_types):
        """
        Args:
            spectrogram_types: List of spectrogram types to generate.
                             Defaults to ["mel"] if not specified.
        """
        self.chain_names = ["LL", "LP", "RP", "RR", "CZ"]
        self.electrodes_per_chain = [
            ["Fp1", "F7", "T3", "T5", "O1"],
            ["Fp1", "F3", "C3", "P3", "O1"],
            ["Fp2", "F8", "T4", "T6", "O2"],
            ["Fp2", "F4", "C4", "P4", "O2"],
            ["Fz", "Cz", "Pz"],
        ]
        self._result_dim = (128, 256)  # height, width
        self._sampling_rate = 200  # Hz
        self._frequency_range = (0, 25)  # Hz

        if spectrogram_types is None:
            spectrogram_types = ["mel"]

        self._spectrogram_types = spectrogram_types

        # validate spectrogram types
        supported_types = ["mel", "stft", "cwt"]
        for spec_type in self._spectrogram_types:
            if spec_type not in supported_types:
                raise ValueError(f"Unsupported spectrogram type: {spec_type}. Supported types: {supported_types}")

    def generate(self, eeg_data):
        """
        Generate spectrograms from 50-second EEG data.

        Args:
            eeg_data: DataFrame containing 50 seconds (10000 samples) of EEG data

        Returns:
            Dictionary with entry for each spectrogram type with numpy array of shape (height, width, channels)
        """
        if len(eeg_data) != 10000:
            raise ValueError(f"Expected 10000 samples (50 seconds), got {len(eeg_data)}")

        results = {}

        for spec_type in self._spectrogram_types:
            results[spec_type] = self._generate_spectrogram(eeg_data, spec_type)

        return results

    def _generate_spectrogram(self, eeg_data, spec_type):
        """
        Generate spectrogram from EEG data.

        Args:
            eeg_data: DataFrame containing EEG data
            spec_type: Type of spectrogram to generate

        Returns:
            Numpy array of shape (128, 256, 5) representing the spectrogram
        """
        # output image has height=128, width=256, channels=5 (for 5 chains)
        spectrogram = np.zeros((self._result_dim[0], self._result_dim[1], len(self.chain_names)), dtype="float32")

        for chain_index in range(len(self.chain_names)):
            electrodes = self.electrodes_per_chain[chain_index]
            pair_difference_num = len(electrodes) - 1

            # for each pair of electrodes in this chain, compute spectrogram of their difference
            for pair_difference_index in range(pair_difference_num):
                electrode1 = electrodes[pair_difference_index]
                electrode2 = electrodes[pair_difference_index + 1]

                pair_difference = eeg_data[electrode1] - eeg_data[electrode2]
                pair_difference = np.array(pair_difference, dtype="float32")
                pair_difference = fill_nan_with_mean(pair_difference)

                if spec_type == "mel":
                    spec_db = self._create_mel_spectrogram(pair_difference)
                elif spec_type == "stft":
                    spec_db = self._create_stft_spectrogram(pair_difference)
                elif spec_type == "cwt":
                    spec_db = self._create_cwt_spectrogram(pair_difference)
                else:
                    raise ValueError(f"Unsupported spectrogram type: {spec_type}")

                spectrogram[:, :, chain_index] += spec_db

            # average the spectrogram differences for this chain to get the final spectrogram
            spectrogram[:, :, chain_index] /= pair_difference_num

        return spectrogram

    def _create_mel_spectrogram(self, pair_difference):
        """
        Create a mel spectrogram from electrode pair difference data.

        Args:
            pair_difference: 1D array of electrode difference values

        Returns:
            2D numpy array representing the mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=pair_difference,
            sr=self._sampling_rate,  # sampling frequency is 200Hz
            hop_length=len(pair_difference) // self._result_dim[1],  # produces image with width = len(x)/hop_length
            n_fft=1024,  # controls vertical resolution and quality of spectrogram
            n_mels=self._result_dim[0],  # number of mel bands, corresponds to height of spectrogram
            fmin=self._frequency_range[0],  # min frequency
            fmax=self._frequency_range[0],  # max frequency
            win_length=self._result_dim[0],  # window size, controls horizontal resolution and quality of spectrogram
        )

        # convert from power to db scale
        # ref=np.max scales to max value of 0dB, with everything else negative
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # normalize to [-1, 1]
        mel_spec_db = self._normalize_spectrogram(mel_spec_db)

        # cut to width 256
        mel_spec_db = mel_spec_db.astype(np.float32)[:, : self._result_dim[1]]

        return mel_spec_db

    def _create_stft_spectrogram(self, pair_difference):
        """
        Create an STFT spectrogram from electrode pair difference data.

        Args:
            pair_difference: 1D array of electrode difference values

        Returns:
            2D numpy array of shape (128, 256) representing the STFT spectrogram
        """
        n_fft = 1024
        stft = librosa.stft(
            y=pair_difference,
            n_fft=n_fft,  # controls vertical resolution and quality of spectrogram
            hop_length=len(pair_difference) // self._result_dim[1],  # ensures output width of 256 time frames
            win_length=self._result_dim[0],  # window length for temporal localization
            window="hann",  # Hann window reduces spectral leakage (is the default)
            center=True,  # pad signal to center frames
        )

        # get magnitude from complex numbers
        magnitude = np.abs(stft)

        # extract only low-frequency bins (0-25 Hz)
        # with sampling_rate=200 and n_fft=1024, frequency resolution is 200/1024 ≈ 0.195 Hz per bin
        # taking first 128 bins covers approximately 0-25 Hz, capturing the EEG range of interest
        hz_per_bin = self._sampling_rate / n_fft
        num_bins = int(self._frequency_range[1] / hz_per_bin)
        magnitude = magnitude[:num_bins, :]

        # convert to dB scale
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

        # normalize to [-1, 1]
        magnitude_db = self._normalize_spectrogram(magnitude_db)

        # ensure exact width of 256
        magnitude_db = magnitude_db.astype(np.float32)[:, : self._result_dim[1]]

        return magnitude_db

    def _create_cwt_spectrogram(self, pair_difference):
        """
        Create a CWT (Continuous Wavelet Transform) spectrogram from electrode pair difference data.

        Args:
            pair_difference: 1D array of electrode difference values (10000 samples)

        Returns:
            2D numpy array of shape (128, 256) representing the CWT spectrogram
        """
        # define 128 frequency scales covering 0.5-25 Hz (EEG range of interest)
        # using logarithmic spacing to get better resolution at lower frequencies
        low_freq = self._frequency_range[0] if self._frequency_range[0] > 0 else 0.5
        frequencies = np.logspace(np.log10(low_freq), np.log10(self._frequency_range[1]), self._result_dim[0])

        # convert frequencies to scales for the Morlet wavelet
        # the Morlet wavelet has a center frequency that needs to be scaled
        scales = pywt.frequency2scale("morl", frequencies / self._sampling_rate)

        # 'morl' = Morlet wavelet, which is a complex sine wave in a Gaussian envelope
        cwt_matrix, _ = pywt.cwt(
            data=pair_difference, scales=scales, wavelet="morl", sampling_period=1.0 / self._sampling_rate
        )

        # get magnitude from complex-valued CWT coefficients
        magnitude = np.abs(cwt_matrix)

        # reverse frequency axis so low frequencies are at bottom
        magnitude = np.flipud(magnitude)

        # downsample time axis to 256 frames for consistency with other spectrograms
        # use interpolation to smoothly reduce from 10000 to 256 time points
        time_indices = np.linspace(0, magnitude.shape[1] - 1, self._result_dim[1])
        magnitude_resampled = np.zeros((self._result_dim[0], self._result_dim[1]), dtype=np.float32)

        for freq_idx in range(self._result_dim[0]):
            magnitude_resampled[freq_idx, :] = np.interp(
                time_indices, np.arange(magnitude.shape[1]), magnitude[freq_idx, :]
            )

        # convert to dB scale (amplitude, not power!)
        magnitude_db = librosa.amplitude_to_db(magnitude_resampled, ref=np.max)

        # normalize to [-1, 1]
        magnitude_db = self._normalize_spectrogram(magnitude_db)

        return magnitude_db.astype(np.float32)

    def _normalize_spectrogram(self, spectrogram):
        """
        Normalize spectrogram to [-1, 1] range.

        Args:
            spectrogram: 2D numpy array representing the spectrogram

        Returns:
            Normalized spectrogram
        """
        db_min, db_max = -80, 0
        spectrogram = np.clip(spectrogram, db_min, db_max)
        spectrogram = 2 * (spectrogram - db_min) / (db_max - db_min) - 1
        return spectrogram
