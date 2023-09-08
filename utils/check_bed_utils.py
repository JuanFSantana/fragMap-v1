from dataclasses import dataclass
import pandas as pd
import sys
from typing import Tuple


@dataclass
class BedFile:
    """
    A class to represent a BedFile.

    ...

    Attributes
    ----------
    bfile : str
        The path to the bed file.

    Methods
    -------
    check_regions() -> Tuple[int, int]:
        Checks if the bed file follows certain conditions and returns the shape of the DataFrame.
    """

    bfile: str

    def check_regions(self) -> Tuple[int, int]:
        """
        Checks if the bed file follows certain conditions.

        The conditions checked are:
            1. The bed file should not have more than 6 columns.
            2. For each row, the second column's value should be greater than the first column's value.
            3. All regions should be of the same size and even.
            4. Column 3 should have unique labels.

        If the conditions are not met, the program exits with an error message.
        If the conditions are met, it returns the shape of the DataFrame generated from the bed file.

        Returns
        -------
        Tuple[int, int]
            The shape of the DataFrame if the conditions are met.
        """
        df = pd.read_csv(self.bfile, sep="\t", header=None)
        if df.shape[1] > 6:
            return sys.exit("Your bed file has more than 6 columns. Exiting")
        elif any(df[2] - df[1] < 0):
            return sys.exit("Coordinate in column 2 should be greater than coordinate in column 1. Exiting")
        elif len(set(df[2] - df[1])) > 1:
            return sys.exit("All regions should be of the same size. Exiting")
        elif list(set(df[2] - df[1]))[0] % 2 != 0:
            return sys.exit("All regions should be of even length. Exiting")
        elif len(set(df[3])) != len(df[3]):
            return sys.exit("All regions should have a unique identifier in column 4. Exiting")
        return df.shape[0], df.shape[1]
