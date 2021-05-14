"""
@author: Johanna Rahm
Research group Heilemann
Institute for Physical and Theoretical Chemistry, Goethe University Frankfurt a.M.

Calculate k nearest neighbors & display them as k-distance graphs.
"""

import os
import sys
import configparser
import math
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from kneed import KneeLocator  # https://github.com/arvkevi/kneed


class IncorrectConfigException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class GridkNNSearch:
    """
    The computing time of the nearest neighbor analysis is accelerated by a grid, which lies over the space
    of the localizations, thereby creating subregions. Only neighbours in adjacent subregions are accepted as
    potential nearest neighbor candidates for a center.
    """
    def __init__(self, centers, neighbors, number_subregions, pixel_size, n_px_x, n_px_y):
        self.pixel_size = pixel_size  # 158 nm
        self.number_subregions = number_subregions  # e.g. ~ number of neighbor points
        self.border_length = (n_px_x*pixel_size/number_subregions, n_px_y*pixel_size/number_subregions)  # length of subregion border
        self.neighbors = neighbors  # list of list of neighbors xy-locations
        self.centers = centers  # xy-locations of centers
        self.grid_neighbors = self.create_grid(number_subregions)  # subregions linked to their points (xy-locs)
        self.center_is_neighbor = self.center_is_neighbor()  # list of booleans if center list equals a neighbor list

    def center_is_neighbor(self):
        """
        Check if the center list is equal with each of the neighbor lists.
        :return: List of booleans if center list equals neighbor lists
        """
        center_is_neighbor = []
        for neighbor_list in self.neighbors:
            is_same = True if np.array_equal(self.centers, neighbor_list) else False
            center_is_neighbor.append(is_same)
        return center_is_neighbor

    def create_grid(self, number_subregions):
        """
        Divide the space into subregions that store the localizations of neighbors in their subspace.
        :param number_subregions: Total number of subregions.
        :return: Dict of key = subregion index & value = list of neighbors, their original indices and reference
        to neighbor list.
        """
        # dict with key = XY and value = []
        grid_neighbors = {}
        for x in range(number_subregions):
            for y in range(number_subregions):
                grid_neighbors.update({(x, y): []})
        # store neighbor xy-locations in corresponding subregion
        for c, neighbor_list in enumerate(self.neighbors):
            for i, neighbor in enumerate(neighbor_list):
                subregion = self.point_to_subregion(neighbor)
                # i = idx in list, c = idx of neighbor list (FGFR1 -> 0, FGFR2 -> 1 ...)
                grid_neighbors[subregion].append((neighbor, i, c))
        return grid_neighbors

    def get_k_nn_distances(self, k):
        """
        Calculate the distance to the k nearest neighbors for each center.
        :return: k Nearest neighbor distances, sorted by center, sorted by k.
        """
        k_nn_distances = []
        k_nn_distances_k_sorted = [[] for _ in range(k)]
        with tqdm(total=len(self.centers)) as pbar:
            for center_idx, center in enumerate(self.centers):
                subregion = self.point_to_subregion(center)
                search_area = 0
                found_neighbor = False
                while not found_neighbor:
                    # check if neighbors are in search_area range & if the neighbor is not the particle itself
                    # if no neighbors are found, increase the search_area
                    # for each valid sub region get the xy-localizations
                    neighbors = list(
                        map(lambda x: self.grid_neighbors[x], self.get_valid_sub_regions(subregion, search_area)))
                    # merge elements of sublists to one list
                    neighbors_conc = [j for i in neighbors for j in i]
                    # delete the element in neighbors that equals center
                    for c, neighbor in enumerate(neighbors_conc):
                        if np.array_equal(neighbor[0], center) and neighbor[1] == center_idx:
                            neighbors_conc.pop(c)
                    # set appropriate search area (area that found neighbor +1)
                    if len(neighbors_conc) >= k:
                        search_area += 1
                        found_neighbor = True
                    else:
                        search_area += 1
                valid_sub_regions = self.get_valid_sub_regions(subregion, search_area)
                # get all potential nearest neighbor candidates from the valid subregions
                neighbors_in_area = list(map(lambda x: self.grid_neighbors[x], valid_sub_regions))
                neighbors_in_area_conc = [j for i in neighbors_in_area for j in i]
                for c, neighbor in enumerate(neighbors_in_area_conc):
                    if np.array_equal(neighbor[0], center) and neighbor[1] == center_idx:
                        neighbors_in_area_conc.pop(c)
                # calc k nearest neighbors for a center point, the lst idx referes to k.
                k_nn_distance = self.calc_k_min_distances(center, neighbors_in_area_conc, k)
                for c, i in enumerate(k_nn_distance):
                    k_nn_distances_k_sorted[c].append(i)
                k_nn_distances.append(k_nn_distance)
                pbar.update(1)
        return k_nn_distances, k_nn_distances_k_sorted

    def calc_k_min_distances(self, center, neighbor_candidates, k):
        """
        Calculate the euclidean distance of candidates in a list and return the k nearest neighbor distances.
        :param center: Target center.
        :param neighbor_candidates: Potential k nearest neighbors.
        :return: K nearest neighbor distances.
        """
        distances = []
        for idx in range(len(neighbor_candidates)):
            distance = np.linalg.norm(center - neighbor_candidates[idx][0])
            distances.append(distance)
        distances = sorted(distances)[:k]
        return distances

    def point_to_subregion(self, point):
        """
        Sort a xy-localization to a subregion.
        :param point: xy-localization.
        :return: (x, y) refers to the subregion indices.
        """
        x = int(np.floor(point[0] / self.border_length[0]))
        y = int(np.floor(point[1] / self.border_length[1]))
        # this only happens if the localization is directly at the border of the measurement space.
        if x == self.number_subregions:
            x -= 1
        if y == self.number_subregions:
            y -= 1
        return x, y

    def get_valid_sub_regions(self, center, search_area):
        """
        From a center subregion get all valid subregions within the appropriate search area.
        :param center: Subregion indices.
        :param search_area: Valid deviations of center subregion-id.
        :return: List of valid subregion-ids.
        """
        valid_sub_regions = []
        for x in range(center[0] - search_area, center[0] + search_area + 1):
            for y in range(center[1] - search_area, center[1] + search_area + 1):
                if x in range(self.number_subregions) and y in range(self.number_subregions):
                    valid_sub_regions.append((x, y))
        return valid_sub_regions


def get_xy_locs(h5_path, pixel_size):
    h5_file = h5py.File(h5_path, "r")
    xy_stack = np.column_stack((h5_file["locs"]["x"]*pixel_size, h5_file["locs"]["y"]*pixel_size))
    #xy_stack = xy_stack[:int(len(xy_stack)*0.1)]
    return xy_stack


def run_k_nn(xy_stack, pixel_size, camera_size, max_k):
    knearest_neighbor = GridkNNSearch(xy_stack, [list(xy_stack)],
                                      int(np.floor(math.sqrt(len(xy_stack)))), pixel_size, camera_size, camera_size)
    knn_distances, _ = knearest_neighbor.get_k_nn_distances(max_k)
    distances_sorted = []
    for c in range(max_k):
        distances_sorted.append(np.sort([i[c] for i in knn_distances])[::-1])
    idx = [i for i in range(1, len(knn_distances)+1)]
    return distances_sorted, idx


def plot_graph(filename, knn_distances, idx, max_k, calc_knee_point, kns, save_dir, save_plot):
    for k, kn, distances in zip(range(1, max_k+1), kns, knn_distances):
        plt.figure()
        plt.plot(idx, distances, label="distances")
        if calc_knee_point.lower() == "yes":
            plt.plot([min(idx), max(idx)], [kn, kn], "--", label="knee-point")
            plt.title(str(k)+". distances [nm], knee-point="+str(kn)+" nm")
        else:
            plt.title(str(k) + ". distances [nm]")
        plt.xlabel(filename)
        plt.ylabel("distance [nm]")
        plt.legend()
        if save_plot.lower() == "yes":
            plt.savefig(save_dir + "\\" + filename + "_" + str(k) + "_distance_graph.png")
        plt.close()


def get_knee_point(knn_distances, idx, filter_width):
    kns = []
    for distances in knn_distances:
        # low pass filtering (linear)
        if filter_width <= 0:
            filter_width = 1
        filtered_distances = [sum(distances[i:i + filter_width]) / filter_width for i in range(0, len(distances) - filter_width)]
        # normalize x and y
        max_y = max(filtered_distances)
        max_x = len(idx) - filter_width
        filtered_distances = [y / max_y for y in filtered_distances]
        idx_n = [x / max_x for x in idx[:-filter_width]]

        kn = KneeLocator(idx_n, filtered_distances, curve="convex", direction="decreasing").knee * max_y
        kn = np.round(kn * 10000) / 10000
        kns.append(kn)
    return kns


def save_knee_points_csv(h5_path, kns, save_dir):
    out_file_name = save_dir + "\\" + os.path.splitext(os.path.basename(h5_path))[0] + "_knee_points.csv"
    header = "k, knee point [nm]"
    data = np.zeros(np.array(kns).size, dtype=[("col1", int), ("col2", float)])
    data["col1"] = [i for i in range(1, len(kns)+1)]
    data["col2"] = kns
    np.savetxt(out_file_name, X=data, fmt=("%d", "%.4e"), header=header, delimiter=",")


def save_distances_csv(h5_path, knn_distances, max_k, save_dir):
    out_file_name = save_dir + "\\" + os.path.splitext(os.path.basename(h5_path))[0] + "_k_distances.csv"

    header = "index,"
    fmt = ("%d",)
    for k in range(1, max_k+1):
        header += str(k) + " nearest neighbor [nm],"
        fmt += ("%.4e",)

    dtype = [("idx", int)]
    for k in range(1, max_k + 1):
        dtype.append(("col"+str(k), float))

    data = np.zeros(np.array(knn_distances[0]).size, dtype=dtype)
    data["idx"] = [i for i in range(1, len(knn_distances[0])+1)]
    for k in range(1, max_k+1):
        data["col"+str(k)] = knn_distances[k-1]

    np.savetxt(out_file_name, X=data, fmt=fmt, header=header[:-1], delimiter=",")


def main(cfg_path):
    config = configparser.ConfigParser()
    config.sections()
    config.read(cfg_path)

    h5_paths = []
    try:
        if len([key for key in config["INPUT_FILES"]]):
            for key in config["INPUT_FILES"]:
                h5_paths.append(config["INPUT_FILES"][key])
        else:
            raise IncorrectConfigException("No input path defined in config.")
    except KeyError:
        raise IncorrectConfigException("Section INPUT_FILES missing in config.")

    try:
        pixel_size = int(config["PARAMETERS"]["pixel_size"])
    except KeyError:
        raise IncorrectConfigException("Parameter pixel_size missing in config.")

    try:
        camera_size = int(config["PARAMETERS"]["camera_size"])
    except KeyError:
        raise IncorrectConfigException("Parameter camera_size missing in config.")

    try:
        max_k = int(config["PARAMETERS"]["max_k"])
    except KeyError:
        raise IncorrectConfigException("Parameter max_k missing in config.")

    try:
        calc_knee_point = config["PARAMETERS"]["calc_knee_point"]
    except KeyError:
        raise IncorrectConfigException("Parameter calc_knee_point missing in config.")

    try:
        low_pass_width = int(config["PARAMETERS"]["low_pass_width"])
    except KeyError:
        raise IncorrectConfigException("Parameter low_pass_width missing in config.")

    try:
        save_plots = config["PARAMETERS"]["save_plots"]
    except KeyError:
        raise IncorrectConfigException("Parameter save_plots missing in config.")

    try:
        save_dir = config["SAVE_DIR"]["save_dir"]
    except KeyError:
        raise IncorrectConfigException("No save directory defined in config.")

    for h5_path in h5_paths:
        xy_locs = get_xy_locs(h5_path, pixel_size)
        distances_ks, idx = run_k_nn(xy_locs, pixel_size, camera_size, max_k)
        kns = get_knee_point(distances_ks, idx, low_pass_width)
        plot_graph(os.path.splitext(os.path.basename(h5_path))[0], distances_ks, idx, max_k, calc_knee_point, kns,
                   save_dir, save_plots)
        save_distances_csv(h5_path, distances_ks, max_k, save_dir)
        if calc_knee_point.lower() == "yes":
            save_knee_points_csv(h5_path, kns, save_dir)

    print("k distances successfully calculated.")


if __name__ == "__main__":
    try:
        cfg_path = sys.argv[1]
        main(cfg_path)
    except IndexError:
        print("Usage: python k_distance_graph.py your_config_file.ini")
