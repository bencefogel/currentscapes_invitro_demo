import pandas as pd
import numpy as np

from currentscape_calculator.partitioning_algorithm import partition_iax

class CurrentscapeCalculator:
    def __init__(self, target: str, partitioning_strategy: str, regions_list_directory: str) -> None:
        self.target = target
        self.partitioning_strategy = partitioning_strategy
        self.regions_list_directory = regions_list_directory,

    def calculate_currentscape(self, iax, im, taxis, tmin, tmax):
        # Load data for the given pair of files
        df_iax = pd.read_csv(iax, index_col=[0,1])
        df_iax.columns = df_iax.columns.astype(int)
        df_im = pd.read_csv(im, index_col=[0,1])
        df_im.columns = df_im.columns.astype(int)

        if self.partitioning_strategy == 'type':
            df_im.sort_index(axis=0, level=(0, 1), inplace=True)

        # if no timepoints is selected, we perform the partitioning on the whole dataframe
        if tmin and tmax is None:
            t_max = df_im.shape[1]
            segment_indexes = list(range(t_max))
        else:
            segment_indexes = np.flatnonzero((taxis > tmin) & (taxis < tmax))

        # Perform the partitioning
        im_part_pos, im_part_neg = partition_iax(df_im, df_iax, timepoints=segment_indexes, target=self.target,
                                                 partition_by=self.partitioning_strategy,
                                                 regions_list_directory=self.regions_list_directory)
        return im_part_pos, im_part_neg

    def load_df(self, index_fname: str, values_fname: str):
        """
        Loads a DataFrame from a CSV file containing a multiindex and a NumPy file containing the corresponding values.

        Parameters:
            index_fname (str): The file path to the CSV file containing the multiindex data.
            values_fname (str): The file path to the .npy file containing the array of values.

        Returns:
            pd.DataFrame: A pandas DataFrame constructed using the multiindex from the CSV file and the values from the .npy file.
        """
        index = pd.read_csv(index_fname)
        values = np.load(values_fname)

        multiindex = pd.MultiIndex.from_frame(index)
        df = pd.DataFrame(data=values, index=multiindex)
        return df
