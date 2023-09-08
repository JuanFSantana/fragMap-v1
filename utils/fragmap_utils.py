import subprocess
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ProcessedData:
    """
    A class used to represent the processed data in a structured way.

    Attributes
    ----------
    analyzed_data : pd.DataFrame
        The data after it has been analyzed, defaults to None.
    to_concat : list of pd.DataFrame
        A list of dataframes that are intended to be concatenated, defaults to None.
    concat_data : pd.DataFrame
        The result of concatenating dataframes in the to_concat list, defaults to None.
    final_matrix : pd.DataFrame
        The final matrix obtained after processing the data, defaults to None.
    array_to_image : np.ndarray
        A NumPy array that represents an image, defaults to None.
    """

    analyzed_data: pd.DataFrame = None
    to_concat: List[pd.DataFrame] = None
    concat_data: pd.DataFrame = None
    final_matrix: pd.DataFrame = None
    array_to_image: np.ndarray = None


@dataclass
class FragMap:
    """
    A class used to represent a FragMap.

    Attributes
    ----------
    image_name : str
        The name of the image to be generated from the FragMap.
    regions : str
        Path to bed file of the regions of the genome being analyzed.
    fragments : str
        Path to bed file with the fragments/reads of the genome being analyzed.
    fragment_size_left : int
        The size of the left fragment.
    fragment_size_right : int
        The size of the right fragment.
    correction_factor : float
        A correction factor to be applied to the data, defaults to 1.0.
    y_axis : int
        The size of the y-axis in the image, defaults to 1.
    x_axis : float
        The size of the x-axis in the image, defaults to 1.0.
    centers : bool
        Whether to analyze the centers of the fragments or not, defaults to False.
    bedtools_output : str
        The output from bedtools, defaults to None.
    region_size : int
        The size of the region being analyzed, defaults to None.
    processed_data : ProcessedData
        An instance of the ProcessedData class, initialized in __post_init__.
    """

    image_name: str
    regions: str
    fragments: str
    fragment_size_left: int
    fragment_size_right: int
    correction_factor: float = 1.0
    y_axis: int = 1
    x_axis: float = 1.0
    centers: bool = False
    bedtools_output: str = None
    region_size: int = None
    processed_data: ProcessedData = field(init=False)

    def __post_init__(self):
        """
        Performs additional initialization after the dataclass has been initialized.

        This method is automatically called after the class is fully initialized.
        It is used to set up the 'processed_data' attribute and calculate the region size.

        Attributes
        ----------
        processed_data : ProcessedData
            An instance of the ProcessedData class.
        region_size : int
            The size of the region being analyzed. This is calculated based on the input regions file.
        """
        self.processed_data = ProcessedData()
        # Read just the first row of the file
        df = pd.read_csv(self.regions, sep="\t", header=None, nrows=1)
        self.region_size = df[2][0] - df[1][0]

    def _run_bedtools(self) -> None:
        """
        Runs bedtools intersect on the regions and fragments files, storing the output in a temporary file. If 'bedtools_output' is already set, this method does nothing.
        """
        if self.bedtools_output is None:
            random_name = str(uuid.uuid4())
            temp_data_bedtools = Path(Path.cwd(), random_name + ".bed")
            subprocess.call(
                " ".join(
                    [
                        "bedtools intersect -a",
                        self.regions,
                        "-b",
                        self.fragments,
                        "-wa",
                        "-wb",
                        ">",
                        str(temp_data_bedtools),
                    ]
                ),
                shell=True,
            )
            self.bedtools_output = temp_data_bedtools

    def get_data(self) -> pd.DataFrame:
        """
        Processes the bedtools output and formats it into a usable form. If no bedtools output is available, it is generated. The method also filters by fragment size and computes new coordinates based on certain conditions.
        Returns
        -------
        pd.DataFrame
            The DataFrame containing processed bedtools output.
        """
        if self.bedtools_output is None:
            self._run_bedtools()
            try:
                cols = [1, 2, 3, 5, 7, 8, 11]
                df_bedtools = pd.read_csv(
                    self.bedtools_output, sep="\t", header=None, usecols=cols
                )

                # Fragment size col
                df_bedtools["Fragment_size"] = (df_bedtools[8] - df_bedtools[7]) + 1

                if self.centers == False:
                    # Convert intervals where one or both of the sides are smaller or larger than the region into the region limits
                    df_bedtools["New_read_start"] = np.where(
                        df_bedtools[7] < df_bedtools[1], df_bedtools[1], df_bedtools[7]
                    )
                    df_bedtools["New_read_end"] = np.where(
                        df_bedtools[8] > df_bedtools[2], df_bedtools[2], df_bedtools[8]
                    )
                elif self.centers == True:
                    # Create new intervals for fragment center coordinates:
                    # if center is even, start/end is the same coordinate - the center is counted twice
                    # if the center if odd, the coordinate is split - the center is counted once for each coordinate
                    df_bedtools["New_read_start"] = np.where(
                        (df_bedtools[7] + df_bedtools[8]) % 2 == 0,
                        (df_bedtools[7] + df_bedtools[8]) / 2,
                        ((df_bedtools[7] + df_bedtools[8]) / 2) - 0.5,
                    )
                    df_bedtools["New_read_end"] = np.where(
                        (df_bedtools[7] + df_bedtools[8]) % 2 == 0,
                        (df_bedtools[7] + df_bedtools[8]) / 2,
                        ((df_bedtools[7] + df_bedtools[8]) / 2) + 0.5,
                    )

                # Convert coordinate to distance from region start/end depending on strand
                df_bedtools["Coor_start"] = np.where(
                    df_bedtools[5] == "+",
                    (df_bedtools["New_read_start"] - df_bedtools[1]),
                    df_bedtools[2] - df_bedtools["New_read_end"],
                )
                df_bedtools["Coor_end"] = np.where(
                    df_bedtools[5] == "+",
                    (df_bedtools["New_read_end"] - df_bedtools[1]),
                    df_bedtools[2] - df_bedtools["New_read_start"],
                )

                # Filter by fragment size
                df_bedtools = df_bedtools.loc[
                    (df_bedtools["Fragment_size"] >= self.fragment_size_left)
                    & (df_bedtools["Fragment_size"] <= self.fragment_size_right)
                ].reset_index(drop=True)

                tmp_name = str(uuid.uuid4())
                temp_data_analyzed = Path(Path.cwd(), tmp_name + ".bed")
                df_bedtools.to_csv(
                    temp_data_analyzed, sep="\t", index=False, header=True
                )
                Path.unlink(self.bedtools_output)

                return temp_data_analyzed

            except pd.errors.EmptyDataError:
                print("No fragments found in the regions")
                sys.exit(1)

    def fragment_size_by_coordinate(
        self, chunk_coor_size_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transforms a DataFrame of fragments and their coordinates into a two-column DataFrame, with each row representing a fragment and the coordinates it overlaps with. This DataFrame is then grouped by fragments and coordinates, and the count is stored as a third column.

        Parameters
        ----------
        chunk_coor_size_df : pd.DataFrame
            A DataFrame containing the coordinates and sizes of various fragments.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the fragment sizes and their corresponding coordinates.
        """
        final = defaultdict(list)
        for indx, row in chunk_coor_size_df.iterrows():
            if row["Coor_start"] == row["Coor_end"]:
                step = 1
            else:
                step = row["Coor_end"] - row["Coor_start"]

            base_frag = np.round(
                np.linspace(row["Coor_start"], row["Coor_end"] - 1, step), 0
            )
            frag_size_associated_base_frag = np.repeat(
                [row["Fragment_size"]], len(base_frag)
            )

            final["Coordinate"].extend(base_frag)
            final["Fragment_size"].extend(frag_size_associated_base_frag)

        final_df = pd.DataFrame(final)
        if final_df.empty:
            pass
        else:
            size_coor_df = (
                final_df.groupby(by=["Fragment_size", "Coordinate"])
                .size()
                .reset_index()
                .rename(columns={0: "Count"})
            )
            return size_coor_df

    def matrix_alingment(self) -> pd.DataFrame:
        """
        Ensures the matrix is complete and correctly shaped by filling missing rows and columns with 0.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the aligned matrix.
        """
        # Make Fragment_size the row index, Coordinate the col index and Count the values in the matrix
        matrix_unstacked = self.processed_data.concat_data.set_index(
            ["Fragment_size", "Coordinate"]
        ).unstack()

        # If row indices (y axis) are missing, add them and fill in value with 0. Also, any NaN is filled in with a 0
        matrix_corrected_rows = matrix_unstacked.reindex(
            list(
                range(int(self.fragment_size_left), int(self.fragment_size_right) + 1)
            ),
            fill_value=0,
        ).fillna(0)

        # Drop a level of multi index (Count) is order to use the col indexes later on
        matrix_corrected_rows.columns = matrix_corrected_rows.columns.droplevel(0)

        # If col indices (x axis) are missing, add them and fill in value with 0
        matrix_corrected_rows_cols = matrix_corrected_rows.reindex(
            list(range(0, self.region_size)),
            fill_value=0,
            axis="columns",
        )

        # convert back to dataframe
        norm_matrix_corrected_rows_cols = pd.DataFrame(matrix_corrected_rows_cols)

        return norm_matrix_corrected_rows_cols

    def concat_dataframes(self) -> pd.DataFrame:
        """
        Concatenates all DataFrames in the 'to_concat' list of the 'processed_data' attribute, groups by fragment size and coordinate, and sums the counts. If 'correction_factor' is not 1, all counts are multiplied by it.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the concatenated dataframes, grouped by fragment size and coordinate.
        """
        concatenated_df = (
            pd.concat(self.processed_data.to_concat)
            .groupby(by=["Fragment_size", "Coordinate"])
            .sum()
            .reset_index()
            .rename(columns={0: "Count"})
        )
        if self.correction_factor != 1:
            concatenated_df["Count"] = concatenated_df["Count"] * self.correction_factor

        return concatenated_df

    def modifiy_base_per_pixel(self) -> np.ndarray:
        """
        Modifies the 'final_matrix' attribute of the 'processed_data' to match the size requirements. It will average the data if the width is smaller than 1, and repeat the data vertically if the height is greater than 1.

        Returns
        -------
        np.ndarray
            A NumPy array representing the modified matrix.
        """
        # Height and width aspect
        if self.y_axis >= 1 and self.x_axis == 1:
            vertically_repeated = self.processed_data.final_matrix.reindex(
                self.processed_data.final_matrix.index.repeat(self.y_axis)
            )

        elif self.y_axis >= 1 and self.x_axis < 1:
            # average first
            # pixels per base
            width_rolling_avg = int(1 / self.x_axis)
            # rolling average and select rows containing the average window HORIZONTALLY
            df_matrix_width_avg = (
                self.processed_data.final_matrix.rolling(width_rolling_avg, axis=1)
                .mean()
                .dropna(axis=1, how="any")
            )
            avg_matrix = df_matrix_width_avg[
                df_matrix_width_avg.columns[::width_rolling_avg]
            ]
            # repeat array vertically
            vertically_repeated = avg_matrix.reindex(
                avg_matrix.index.repeat(self.y_axis)
            )

        return vertically_repeated.to_numpy()
