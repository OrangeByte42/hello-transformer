import pickle
import numpy as np
from typing import Any


def load_obj_by_pickle(save_path: str) -> Any:
    """Load an object from a pickle file.
    @param save_path: Path to the pickle file
    @return: Loaded object
    """
    with open(save_path, 'rb') as f:
        obj: Any = pickle.load(f)
    return obj

def set_axes(axes: Any, title: str,
                xlabel: str, xlim: list, xticks: np.ndarray,
                ylabel: str, ylim: list, yticks: np.ndarray,
                legend_loc: str, grid: bool) -> None:
    """Set the properties of the axes.
    @param axes: The axes to set the properties for
    @param title: Title of the axes
    @param xlabel: Label for the x-axis
    @param xlim: Limits for the x-axis
    @param xticks: Ticks for the x-axis
    @param ylabel: Label for the y-axis
    @param ylim: Limits for the y-axis
    @param yticks: Ticks for the y-axis
    @param legend_loc: Location of the legend
    @param grid: Whether to show grid lines
    @return: None
    """
    # title
    axes.set_title(title)

    # x-axis
    axes.set_xlabel(xlabel)
    if xlim is not None: axes.set_xlim(xlim)
    axes.set_xticks(xticks)

    # y-axis
    axes.set_ylabel(ylabel)
    axes.set_ylim(ylim)
    axes.set_yticks(yticks)

    # legend & grid
    axes.grid(grid)
    axes.legend(loc=legend_loc)



