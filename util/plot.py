from typing import Any, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np


def plot_images(imgs: List[Union[np.ndarray, Tuple[np.ndarray, str, Any]]], title='', **fig_kw: Any):
    ''' 
    Plot images in a single row.

    :param imgs: List of images, where each entry is either:
                 - A numpy array representing an image
                 - A tuple (image, title, colormap) where 
                   `title` is a string
                   `colormap` is a cmap for matplotlib (optional)
    :param title: Title of the entire figure (optional)
    :param fig_kw: Additional figure keyword arguments for `plt.subplots()`
    :return: The figure and axes objects.
    '''
    fig, axes = plt.subplots(1, len(imgs), **fig_kw)

    if len(imgs) == 1:
        axes = [axes]

    for i, img in enumerate(imgs):

        if(isinstance(img, tuple)):
            if len(img) == 2:
                img, plot_title = img
                cmap = None
            else: 
                img, plot_title, cmap = img
                
            axes[i].imshow(img, cmap)
            axes[i].set_title(plot_title)

        else:
            axes[i].imshow(img)

    if title:
        fig.suptitle(title)

    # plt.show()
    return fig, axes

# Call the function inside a Jupyter Notebook
