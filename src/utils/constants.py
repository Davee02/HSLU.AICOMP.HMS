from pathlib import Path


class Constants:
    KAGGLE_DATA_BASE_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification")
    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    EEG_FEATURES = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
        "EKG",
        "Fz",
        "Cz",
        "Pz",
    ]

    EEG_ID_COL = "eeg_id"
    SEED = 42
