import concurrent.futures
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


class HeatmapAnalysisType(Enum):
    """
    Enum representing the type of heatmap analysis.

    Attributes
    ----------
    FRAGMAP : str
        Fragmap analysis type.
    CLASSIC : str
        Classic analysis type.
    """

    FRAGMAP = "fragmap"
    CLASSIC = "classic"


class HeatmapColorType(Enum):
    """
    Enum representing the color type of a heatmap.

    Attributes
    ----------
    BLACKNWHITE : str
        Black and white color scheme.
    CHIP : str
        Chip color scheme.
    FOLDCHANGE : str
        Fold change color scheme.
    """

    BLACKNWHITE = "Greys"
    CHIP = "jet"
    FOLDCHANGE = "bwr"


@dataclass
class Heatmap:
    """
    A class used to represent a Heatmap.

    Attributes
    ----------
    array : np.ndarray
        The 2-dimensional numpy array that represents the data to be plotted in the heatmap.
    heatmap_type : HeatmapColorType
        The color scheme to be used in the heatmap.
    heatmap_analysis_type : HeatmapAnalysisType
        The type of heatmap analysis to be performed.
    max_value : Union[list, str]
        The maximum value(s) for the heatmap color scale.
    identifier : str
        An identifier for the heatmap.
    y_axis : int
        The size of the y-axis in the heatmap.
    x_axis : float
        The size of the x-axis in the heatmap.
    gamma : float
        The gamma correction value for the heatmap.
    output_directory : str
        The directory where the generated heatmap image will be saved.
    yaxis_min : int, optional
        The minimum value for the y-axis tick labels. Defaults to 0.
    yaxis_max : int, optional
        The maximum value for the y-axis tick labels. Defaults to 0.
    max_color_value : Optional[Union[list, str]], optional
        The maximum color value(s) for the heatmap color scale.

    Methods
    -------
    image() -> None:
        Generates the heatmap image.

    Returns
    -------
    None
    """

    array: np.ndarray
    heatmap_type: HeatmapColorType
    heatmap_analysis_type: HeatmapAnalysisType
    max_value: Union[list, str]
    identifier: str
    y_axis: int
    x_axis: float
    gamma: float
    output_directory: str
    yaxis_min: int = 0
    yaxis_max: int = 0
    max_color_value: Optional[Union[list, str]] = None

    def __post_init__(self):
        """
        Performs additional initialization after the dataclass has been initialized.
        """
        # Call calculate_max_vals and update max_value
        self.max_value = self._calculate_max_vals(self.array)

    @property
    def color(self):
        """
        Returns the color scheme to be used in the heatmap.

        Returns
        -------
        str
            The name of the color scheme.
        """
        return self.heatmap_type.value

    @property
    def analysis(self):
        """
        Returns the type of heatmap analysis to be performed.

        Returns
        -------
        str
            The name of the analysis type.
        """
        return self.heatmap_analysis_type.value

    def _calculate_max_vals(self, matrix: np.ndarray) -> list:
        """
        Calculates the maximum value(s) for the heatmap color scale.

        Parameters
        ----------
        matrix : np.ndarray
            The data matrix for the heatmap.

        Returns
        -------
        list
            The maximum value(s) for the color scale.
        """
        val_list = []
        if (
            self.heatmap_type == HeatmapColorType.CHIP
            or self.heatmap_type == HeatmapColorType.BLACKNWHITE
        ):
            if self.heatmap_analysis_type == HeatmapAnalysisType.CLASSIC:
                nums = 6
                division = 40
                subsuquent_division = 4
            elif self.heatmap_analysis_type == HeatmapAnalysisType.FRAGMAP:
                nums = 3
                division = 2
                subsuquent_division = 2

            if self.max_value == "default":
                for i in range(nums):
                    if i == 0:
                        maxes = int(np.amax(matrix))
                    elif i == 1:
                        maxes = int(maxes / division)
                    else:
                        maxes = int(maxes / subsuquent_division)
                    val_list.append(maxes)
            else:
                return self.max_value

        if self.heatmap_type == HeatmapColorType.FOLDCHANGE:
            if self.max_color_value == "default":
                maxes = np.round(np.amax(matrix), decimals=2)
                for i in range(8):
                    maxes = np.round(maxes / 1.5, decimals=2)
                    if i >= 2:
                        val_list.append(maxes)
            else:
                return self.max_color_value

        return val_list

    def _imaging(self, max_num):
        """
        Generates the heatmap image with the given maximum value.

        Parameters
        ----------
        max_num : int
            The maximum value for the heatmap color scale.

        Returns
        -------
        None
        """
        max_num = int(max_num)
        plt.rcParams["font.size"] = "5"
        plt.rcParams["figure.facecolor"] = "white"

        if self.heatmap_type == HeatmapColorType.FOLDCHANGE:
            vmin = -max_num
        elif (
            self.heatmap_type == HeatmapColorType.BLACKNWHITE
            or self.heatmap_type == HeatmapColorType.CHIP
        ):
            vmin = 0

        fig, ax = plt.subplots(dpi=1200)
        im = ax.imshow(self.array, vmin=vmin, vmax=max_num, cmap=self.color)
        ax.tick_params(direction="out", length=1.8, width=0.3)

        matrix_height, matrix_length = self.array.shape
        steps = int(matrix_length / 10)
        real_xcoor = [i for i in range(0, matrix_length + 1, int(steps))]

        def conversion_y_classic(num_list):
            return [int(i / self.y_axis) for i in num_list]

        def get_ylabels_classic():
            if matrix_height <= 2000:
                ylabels = [i for i in range(int(matrix_height) + 1) if i % 200 == 0]
            else:
                ylabels = [i for i in range(int(matrix_height) + 1) if i % 1000 == 0]
            return ylabels

        def conversion_y_fragmap(num_list):
            true_yticks = [
                (i - int(self.yaxis_min)) * int(self.y_axis) for i in num_list
            ]
            return true_yticks

        def get_ylabels_fragmap():
            if int(self.yaxis_max) - int(self.yaxis_min) <= 500:
                ylabels = [
                    i
                    for i in range(int(self.yaxis_min), int(self.yaxis_max) + 1)
                    if i % 50 == 0
                ]
            else:
                ylabels = [
                    i
                    for i in range(int(self.yaxis_min), int(self.yaxis_max) + 1)
                    if i % 100 == 0
                ]
            return ylabels

        def x_conver(matrix_length, width):
            if width <= 1:
                xlabels_ = [
                    i
                    for i in range(
                        int(-matrix_length / (2 * float(width))),
                        int(matrix_length / (2 * float(width))) + 1,
                        int(steps * (1 / float(width))),
                    )
                ]
            else:
                xlabels_ = [
                    i
                    for i in range(
                        int(-matrix_length / 2 / float(width)),
                        int(matrix_length / 2 / float(width) + 1),
                        int(steps / float(width)),
                    )
                ]
            return xlabels_

        x_axis_converted_nums = x_conver(matrix_length, self.x_axis)
        sorted_x_axis_converted_nums = sorted(x_axis_converted_nums)
        second_min = abs(sorted_x_axis_converted_nums[1])
        if second_min < 1000:
            xlabels = [i if i != 0 else 1 for i in x_axis_converted_nums]
        else:
            xlabels = [
                str(int(i / 1000)) + "k" if i != 0 else 1 for i in x_axis_converted_nums
            ]

        plt.xticks(real_xcoor, xlabels)
        if self.heatmap_analysis_type == HeatmapAnalysisType.CLASSIC:
            ylabels = get_ylabels_classic()
            plt.yticks(ylabels, conversion_y_classic(ylabels))
        elif self.heatmap_analysis_type == HeatmapAnalysisType.FRAGMAP:
            ylabels = get_ylabels_fragmap()
            plt.yticks(conversion_y_fragmap(ylabels), ylabels)

        if self.gamma != 1:
            image_path = Path(
                self.output_directory,
                "-".join(
                    [
                        str(self.identifier),
                        "Max",
                        str(max_num),
                        "X",
                        str(self.x_axis),
                        "Y",
                        str(self.y_axis),
                        "Gamma",
                        str(self.gamma),
                    ]
                )
                + ".png",
            )
        else:
            image_path = Path(
                self.output_directory,
                "-".join(
                    [
                        str(self.identifier),
                        "Max",
                        str(max_num),
                        "X",
                        str(self.x_axis),
                        "Y",
                        str(self.y_axis),
                    ]
                )
                + ".png",
            )

        ax = fig.gca()

        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.tick_params(which="minor", length=0.8, width=0.3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(size=1, width=0.3)
        cbar.outline.set_linewidth(0.1)

        plt.savefig(
            image_path, format="png", facecolor="w", bbox_inches="tight", dpi=1200
        )

        def gammma(x, r):
            """
            From: https://linuxtut.com/en/c3cd5475663c41d9f154/
            Gamma correction y=255*(x/255)
            x Input image
            r Gamma correction coefficient
            """
            y = x / 255
            y = y ** (1 / r)

            return np.uint8(255 * y)

        if self.gamma != 1.0:
            img = Image.open(image_path).convert("RGB")
            arr = np.asarray(img, dtype=float)
            img_gamma = gammma(arr, self.gamma)
            plt.imsave(image_path, img_gamma)

        plt.close()

    def image(self) -> None:
        """
        Generates the heatmap image.

        Returns
        -------
        None
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self._imaging, self.max_value))
