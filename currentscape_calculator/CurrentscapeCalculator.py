import pandas as pd
import numpy as np

from currentscape_calculator.partitioning_algorithm import partition_iax

class CurrentscapeCalculator:
    def __init__(self, target: str, partitioning_strategy: str, regions_list_directory: str) -> None:
        self.target = target
        self.partitioning_strategy = partitioning_strategy
        self.regions_list_directory = regions_list_directory,

    def calculate_currentscape(self, iax_idx, iax_values, im_idx, im_values, timepoints):
        # Load data for the given pair of files
        df_iax = self.load_df(iax_idx, iax_values)
        df_im = self.load_df(im_idx, im_values)

        if self.partitioning_strategy == 'type':
            df_im.sort_index(axis=0, level=(0, 1), inplace=True)

        # if no timepoints is selected, we perform the partitioning on the whole dataframe
        if timepoints is None:
            t_max = df_im.shape[1]
            timepoints = list(range(t_max))

        # Perform the partitioning
        im_part_pos, im_part_neg = partition_iax(df_im, df_iax, timepoints=timepoints, target=self.target,
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

if __name__ == '__main__':
    import os

    output_directory = 'output'
    # partitioning parameters:
    target = 'soma'
    partitioning_strategy = 'type'

    input_directory = os.path.join(output_directory, 'preprocessed')
    regions_list_directory = os.path.join(input_directory, 'regions_list_directory')
    currentscape_calculator = CurrentscapeCalculator(target, partitioning_strategy, regions_list_directory)

    iax_idx = os.path.join(input_directory, 'iax', 'iax_multiindex.csv')
    iax_values = os.path.join(input_directory, 'iax', 'current_values_0.npy')
    im_idx = os.path.join(input_directory, 'im', 'im_multiindex.csv')
    im_values = os.path.join(input_directory, 'im', 'current_values_0.npy')

    im_part_pos, im_part_neg = currentscape_calculator.calculate_currentscape(iax_idx, iax_values, im_idx, im_values,
                                                                              timepoints=None)
