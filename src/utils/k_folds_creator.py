from sklearn.model_selection import StratifiedGroupKFold


class KFoldCreator:
    """
    A class to create stratified group k-folds for cross-validation.
    """

    def __init__(self, n_splits, seed):
        self.n_splits = n_splits
        self.seed = seed

    def create_folds(self, df, stratify_col, group_col):
        """
        Divides the data into `n_splits` folds.
        The group is used to prevent any overlap between the training and validation sets.
        Furthermore, each split is stratified based on the `stratify_col` column to ensure a balanced distribution of classes across folds.
        """
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        folds_df = df.copy()
        folds_df["fold"] = -1
        folds_df.reset_index(drop=True, inplace=True)
        for fold, (_, val_idx) in enumerate(sgkf.split(folds_df, y=folds_df[stratify_col], groups=folds_df[group_col])):
            folds_df.loc[val_idx, "fold"] = fold

        return folds_df
