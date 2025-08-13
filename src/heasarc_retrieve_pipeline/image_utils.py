from collections.abc import Iterable
import numpy as np
import copy
from astropy.table import Table
from astropy.io import fits
from astropy.visualization import hist
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from statsmodels.robust import mad


def image_from_table(table, bins, gaussian_filter_sigma=1.0, correct_zeros=True):
    hist, xbins, ybins = np.histogram2d(table["Y"], table["X"], bins=bins)

    img = gaussian_filter(hist, sigma=gaussian_filter_sigma)

    return xbins, ybins, img.T


def valid_table(table):
    return table[(table["X"] > 0) | (table["Y"] > 0)]


def mask_around_region(table, coord, region_size=30):

    # Note the casting to standard int. Otherwise, it will
    # overflow and give negative numbers
    circle_of_coords = (
        np.array(table["X"] - coord[0]).astype(int) ** 2
        + np.array(table["Y"] - coord[1]).astype(int) ** 2
    )
    return circle_of_coords < region_size**2


def filter_table(table, coord, region_size=30):
    table = valid_table(table)
    table = table[mask_around_region(table, coord, region_size)]

    return table


def filter_table_outside_regions(table, coord_list, region_size=100):
    if len(np.shape(coord_list)) < 2:
        coord_list = np.array([coord_list])
    if not isinstance(region_size, Iterable):
        region_size = np.ones(len(coord_list)) * region_size
    bad = (table["X"] < 0) | (table["Y"] < 0)

    for i, coord in enumerate(coord_list):
        bad = bad | mask_around_region(table, coord, region_size[i])

    table_filt = table[~bad]

    return table_filt


def get_random_fluxes_in_img(table, region_size=30, n_rand=100):
    xmin = np.min(table["X"])
    ymin = np.min(table["Y"])
    xmax = np.max(table["X"])
    ymax = np.max(table["Y"])

    fluxes = []
    for n in range(n_rand):
        x = np.random.uniform(xmin + 3 * region_size, xmax - 3 * region_size)
        y = np.random.uniform(ymin + 3 * region_size, ymax - 3 * region_size)

        table_filt = filter_table(table, [x, y], region_size=region_size)
        fluxes.append(len(table_filt))

    return fluxes


def filter_sources_in_images(eventfile, region_size=30, back_region_size=50):
    hdul = fits.open(eventfile)

    table = Table(copy.deepcopy(hdul[1].data))

    table["ENERGY"] = table["PI"] * 0.04 + 1.6
    good = (
        (table["ENERGY"] >= 3.0)
        & (table["ENERGY"] < 79.0)
        & (table["X"] > 0)
        & (table["Y"] > 0)
    )

    if np.count_nonzero(good) < 20:
        hdul.close()
        return

    table = table[good]
    xmin = np.min(table["Y"])
    ymin = np.min(table["X"])
    xmax = np.max(table["Y"])
    ymax = np.max(table["X"])

    bins = (np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))

    xbins, ybins, img = image_from_table(table, bins)

    dx = np.median(np.diff(xbins))
    dy = np.median(np.diff(ybins))

    # Comparison between image_max and im to find the coordinates of local maxima

    fluxes = get_random_fluxes_in_img(table, region_size=30, n_rand=300)

    median = np.median(fluxes)
    std = mad(fluxes)

    coordinates = peak_local_max(img, min_distance=20)

    coordinates[:, 1] = coordinates[:, 1] * dx + xmin
    coordinates[:, 0] = coordinates[:, 0] * dy + ymin

    fig = plt.figure(eventfile + "0")
    plt.pcolormesh(xbins, ybins, np.log10(img), vmin=np.log10(np.median(img)))
    plt.plot(coordinates[:, 1], coordinates[:, 0], "r.")
    plt.savefig(eventfile.replace(".gz", "").replace(".evt", ".jpg"))
    # plt.close(fig)

    region_fluxes = []
    for i, coord in enumerate(coordinates):
        table_filt = filter_table(table, coord, region_size=region_size)
        flux = len(table_filt)
        region_fluxes.append(flux)

    region_fluxes = np.asarray(region_fluxes)
    order = np.argsort(region_fluxes)
    coordinates = coordinates[order[::-1]]

    for i, coord in enumerate(coordinates):
        table_filt = filter_table(table, coord, region_size=region_size)
        flux = len(table_filt)
        print(median - std, flux, median + std)
        if flux < median + std:
            continue

        hdul[1].data = fits.BinTableHDU(table_filt).data
        hdul.writeto(
            eventfile.replace(".gz", "").replace(".evt", f"_src{i + 1}.evt"),
            overwrite=True,
        )

        x_filt, y_filt, img_filt = image_from_table(
            table_filt, bins, gaussian_filter_sigma=0
        )
        fig = plt.figure(eventfile + f"{i + 1}")
        plt.pcolormesh(x_filt, y_filt, img_filt, vmin=np.median(img))
        plt.savefig(eventfile.replace(".gz", "").replace(".evt", f"_src{i + 1}.jpg"))
        plt.close(fig)

    table_filt = filter_table_outside_regions(
        table, coordinates, region_size=back_region_size
    )

    hdul[1].data = fits.BinTableHDU(table_filt).data
    hdul.writeto(
        eventfile.replace(".gz", "").replace(".evt", f"_back.evt"),
        overwrite=True,
    )
    x_filt, y_filt, img_filt = image_from_table(
        table_filt, bins, gaussian_filter_sigma=0
    )
    fig = plt.figure(eventfile + f"_back")
    plt.pcolormesh(x_filt, y_filt, img_filt, vmin=np.median(img))
    plt.savefig(eventfile.replace(".gz", "").replace(".evt", f"_back.jpg"))
    plt.close(fig)
    hdul.close()
    return True
