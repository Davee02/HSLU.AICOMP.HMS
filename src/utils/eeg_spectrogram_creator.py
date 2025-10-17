import librosa
import numpy as np

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

        if spectrogram_types is None:
            spectrogram_types = ["mel"]

        self._spectrogram_types = spectrogram_types

        # validate spectrogram types
        supported_types = ["mel"]
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
        Generate mel spectrogram from EEG data.

        Args:
            eeg_data: DataFrame containing EEG data
            spec_type: Type of spectrogram to generate

        Returns:
            Numpy array of shape (128, 256, 5) representing the mel spectrogram
        """
        # output image has height=128, width=256, channels=5 (for 5 chains)
        result_width = 256
        spectrogram = np.zeros((128, result_width, len(self.chain_names)), dtype="float32")

        for chain_index in range(len(self.chain_names)):
            electrodes = self.electrodes_per_chain[chain_index]
            pair_difference_num = len(electrodes) - 1

            # for each pair of electrodes in this chain, compute mel spectrogram of their difference
            for pair_difference_index in range(pair_difference_num):
                electrode1 = electrodes[pair_difference_index]
                electrode2 = electrodes[pair_difference_index + 1]

                pair_difference = eeg_data[electrode1] - eeg_data[electrode2]
                pair_difference = np.array(pair_difference, dtype="float32")
                pair_difference = fill_nan_with_mean(pair_difference)

                if spec_type == "mel":
                    spec_db = self._create_mel_spectrogram(pair_difference)
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
        result_width = 256

        mel_spec = librosa.feature.melspectrogram(
            y=pair_difference,
            sr=200,  # sampling frequency is 200Hz
            hop_length=len(pair_difference) // result_width,  # produces image with width = len(x)/hop_length
            n_fft=1024,  # controls vertical resolution and quality of spectrogram
            n_mels=128,  # number of mel bands, corresponds to height of spectrogram
            fmin=0,  # min frequency
            fmax=20,  # max frequency
            win_length=128,  # window size, controls horizontal resolution and quality of spectrogram
        )

        # convert from power to db scale
        # ref=np.max scales to max value of 0dB, with everything else negative
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # normalize to [-1, 1] range based on fixed db min and max
        db_min, db_max = -80, 0
        mel_spec_db = np.clip(mel_spec_db, db_min, db_max)
        mel_spec_db = 2 * (mel_spec_db - db_min) / (db_max - db_min) - 1

        # cut to width 256
        mel_spec_db = mel_spec_db.astype(np.float32)[:, :result_width]

        return mel_spec_db
