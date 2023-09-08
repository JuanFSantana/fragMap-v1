import argparse
import concurrent.futures
import multiprocessing as mp
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.check_bed_utils import BedFile
from utils.fragmap_utils import FragMap
from utils.heatmap_utils import Heatmap, HeatmapAnalysisType, HeatmapColorType


def imaging(heatmap_obj: Heatmap):
    """
    Creates an image representation of a heatmap using the input heatmap object's image method.

    Parameters:
    heatmap_obj (Heatmap): An instance of a Heatmap object.

    Returns:
    Object: Returns the image object created by the heatmap's image method.
    """
    print(f"Creating FragMap for {heatmap_obj.identifier}...")
    return heatmap_obj.image()


def modifying_matrix(fragmap_obj: FragMap) -> np.ndarray:
    """
    Modifies the base matrix of the input FragMap object using its modify_base_per_pixel method.
    This could involve averaging or repeating values in the matrix.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    np.ndarray: The modified matrix.
    """
    return fragmap_obj.modifiy_base_per_pixel()


def fillin_matrix_gaps(fragmap_obj: FragMap) -> pd.DataFrame:
    """
    Fills in the gaps in the FragMap object's matrix where there are no reads for a given fragment size,
    using the object's matrix_alingment method.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    pd.DataFrame: The filled-in matrix as a DataFrame.
    """
    return fragmap_obj.matrix_alingment()


def combined_dataframes(fragmap_obj: FragMap) -> pd.DataFrame:
    """
    Concatenates dataframes for all regions in the input FragMap object using its concat_dataframes method.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    pd.DataFrame: The concatenated DataFrame.
    """
    return fragmap_obj.concat_dataframes()


def col_3_matrix_fragmap(args) -> pd.DataFrame:
    """
    Calculates the number of reads per fragment size and the coordinates of the fragment for the given FragMap object
    and data, using the object's fragment_size_by_coordinate method.

    Parameters:
    args (Tuple): A tuple containing a FragMap object and data.

    Returns:
    pd.DataFrame: DataFrame with calculated number of reads per fragment size and coordinates.
    """
    fragmap_obj, data = args
    return fragmap_obj.fragment_size_by_coordinate(data)


def process_fragmap(fragmap_obj: FragMap) -> str:
    """
    Calculates fragment size and coordinates for each read relative to each genomic region
    for the given FragMap object, using its get_data method.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    str: The path to the processed data.
    """
    return fragmap_obj.get_data()


def get_reads_per_region(fragmap_obj: FragMap) -> List[str]:
    """
    Processes the input FragMap object using a process pool to calculate fragment size and coordinates
    for each read relative to each genomic region. Returns a list of paths to the processed data.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    List[str]: List of paths to the processed data for each region.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_fragmap, fragmap_obj)
        path_to_data = [result for result in results]
    return path_to_data


def making_matrix(fragmap_obj: FragMap, path_files: List[str]) -> None:
    """
    Processes the input FragMap object to create matrices based on the paths provided.
    This includes reading data from the paths, processing data with multiprocessing, and saving results.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.
    path_files (List[str]): List of paths to the data files.

    Returns:
    None
    """
    for i in range(len(fragmap_obj)):
        fragmap_obj[i].processed_data.analyzed_data = path_files[i]
        chunksize = 10000
        cols = [
            7,
            10,
            11,
        ]  # fragment size column = 7, Coor_start column = 10, Coor_end column = 11
        reader = pd.read_csv(
            fragmap_obj[i].processed_data.analyzed_data,
            sep="\t",
            chunksize=chunksize,
            iterator=True,
            usecols=cols,
            dtype={7: int, 10: int, 11: int},
        )

        pool = mp.Pool(mp.cpu_count())
        results = pool.imap_unordered(
            col_3_matrix_fragmap, [(fragmap_obj[i], read) for read in reader]
        )
        pool.close()
        pool.join()
        Path.unlink(fragmap_obj[i].processed_data.analyzed_data)
        fragmap_obj[i].processed_data.to_concat = list(results)


def finalazing_matrix(fragmap_obj: FragMap) -> None:
    """
    Finalizes the matrix of the input FragMap object. This involves concatenating dataframes, filling in gaps,
    and modifying the matrix (such as averaging or repeating values), all performed in a process pool.

    Parameters:
    fragmap_obj (FragMap): An instance of a FragMap object.

    Returns:
    None
    """
    # concat dataframes, fill-in gaps and average/repeat matrix
    with concurrent.futures.ProcessPoolExecutor() as executor:
        concat_results = list(executor.map(combined_dataframes, fragmap_obj))
        for i in range(len(fragmap_obj)):
            fragmap_obj[i].processed_data.concat_data = concat_results[i]

        # make final matrix
        matrix_alingment_results = list(executor.map(fillin_matrix_gaps, fragmap_obj))
        for i in range(len(fragmap_obj)):
            fragmap_obj[i].processed_data.final_matrix = matrix_alingment_results[i]

        # average/repeat matrix
        matrix_repeat_results = list(executor.map(modifying_matrix, fragmap_obj))
        for i in range(len(fragmap_obj)):
            fragmap_obj[i].processed_data.array_to_image = matrix_repeat_results[i]


def making_images(heatmap_obj: Heatmap) -> None:
    """
    Creates an image for each region in the input Heatmap object using a process pool.

    Parameters:
    heatmap_obj (Heatmap): An instance of a Heatmap object.

    Returns:
    None
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(imaging, heatmap_obj))


def parse_args():
    """
    Gets arguments and makes multiple checks
    """

    parser = argparse.ArgumentParser(
        prog="fragmap.py",
        description="Generates fragMaps from specific range of fragment sizes over a chosen genomic interval. Multiple replicates can be compared in a single fragMap\
            A combined fragMap is automatically generated by summing replicates",
    )
    parser.add_argument(
        "regions", type=str, help="Bed file of genomic regions of chosen length"
    )
    parser.add_argument(
        "-f",
        dest="fragments",
        metavar="\b",
        type=str,
        required=True,
        nargs="*",
        help="Bed file of reads/fragments",
    )
    parser.add_argument(
        "-r",
        dest="range",
        metavar="\b",
        type=int,
        nargs=2,
        required=True,
        help="Range of fragment sizes, for exmaple -r 20 400",
    )
    parser.add_argument(
        "-s",
        dest="spikein",
        metavar="\b",
        type=float,
        nargs="*",
        required=True,
        help="Spike-in or correction factors",
    )
    parser.add_argument(
        "-b",
        dest="black",
        metavar="\b",
        default="default",
        help="Sets the chosen value as black, default is largest number in the matrix",
    )
    parser.add_argument(
        "-c",
        dest="centers",
        action="store_true",
        default=False,
        help="If argument is invoked, the output will be a fragMap of centers of fragments",
    )
    parser.add_argument(
        "-y",
        dest="y_axis",
        metavar="\b",
        type=int,
        default=1,
        help="Horizontal lines/bp for each fragment length | Can only be greater or equal than 1",
    )
    parser.add_argument(
        "-x",
        dest="x_axis",
        metavar="\b",
        type=float,
        default=1.0,
        help="Vertical lines/bp for each genomic interval displayed, for example -x 1 is one vertical line/bp; -x 0.1 is one vertical line/10 bp | Can't be greater than 1",
    )
    parser.add_argument(
        "-g",
        dest="gamma",
        metavar="\b",
        type=float,
        default=1.0,
        help="Gamma correction",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        metavar="\b",
        type=str,
        required=True,
        nargs=1,
        help="Path to output",
    )
    parser.add_argument(
        "-n",
        dest="names",
        metavar="\b",
        type=str,
        required=True,
        nargs="*",
        help="Image output names",
    )

    args = parser.parse_args()

    read_file = args.fragments
    size_left, size_right = args.range
    max_val = args.black
    width = args.x_axis
    spikeins = args.spikein
    identifier = args.names

    if float(width) > 1:
        sys.exit("Missing -x argument. x must be int or float less than or equal to 1")

    if len(read_file) != len(spikeins) and len(read_file) != len(identifier):
        sys.exit(
            "The number of bed files, spike-ins or image output names do not match"
        )

    if size_left > size_right:
        sys.exit("Fragment size range is incorrect")

    return args


def main(args):
    regions_to_analyze = args.regions
    read_file = args.fragments
    max_val = args.black
    height = args.y_axis
    width = args.x_axis
    gamma = args.gamma
    output_directory = args.output_dir[0]
    identifier = args.names
    size_left, size_right = args.range
    spikeins = args.spikein
    centers = args.centers

    if max_val != "default":
        try:
            max_val = [int(max_val)]
        except (TypeError, AttributeError, ValueError):
            sys.exit("black value: int or default")

    # check bed file
    if BedFile(regions_to_analyze).check_regions():
        pass

    # instantiate FragMap objects
    fragmap_objs = [
        FragMap(
            regions=regions_to_analyze,
            fragments=read_file[i],
            image_name=identifier[i],
            correction_factor=spikeins[i],
            y_axis=height,
            x_axis=width,
            fragment_size_left=size_left,
            fragment_size_right=size_right,
            centers=centers,
        )
        for i in range(len(read_file))
    ]

    # get data
    print("Getting reads in each region...")
    get_reads_per_region_results = get_reads_per_region(fragmap_objs)
    # make matrix
    print("Counting reads per fragment size...")
    making_matrix(fragmap_objs, get_reads_per_region_results)
    # concat dataframes, fill-in gaps and average/repeat matrix
    print("Finalizing matrix...")
    finalazing_matrix(fragmap_objs)

    # create object for combined data (sum all matrices)
    if len(fragmap_objs) > 1:
        combined_data = [
            FragMap(
                regions=regions_to_analyze,
                fragments=read_file,
                image_name="Combined",
                correction_factor=1,
                y_axis=height,
                x_axis=width,
                fragment_size_left=size_left,
                fragment_size_right=size_right,
                centers=centers,
            )
        ]
        # sum all matrices
        combined_data[0].processed_data.array_to_image = np.sum(
            [i.processed_data.array_to_image for i in fragmap_objs], axis=0
        )
    else:
        combined_data = []

    # create heatmap objects
    heatmap = [
        Heatmap(
            array=obj.processed_data.array_to_image,
            heatmap_type=HeatmapColorType.BLACKNWHITE,
            heatmap_analysis_type=HeatmapAnalysisType.FRAGMAP,
            max_value=max_val,
            identifier=obj.image_name,
            y_axis=height,
            x_axis=width,
            yaxis_min=obj.fragment_size_left,
            yaxis_max=obj.fragment_size_right,
            gamma=gamma,
            output_directory=output_directory,
        )
        for obj in fragmap_objs + combined_data
    ]

    # create heatmaps
    making_images(heatmap)


if __name__ == "__main__":
    args = parse_args()
    main(args)
