from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from pymatgen.core import Structure, Lattice
import random
import os
from scipy.ndimage import gaussian_filter1d
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN
from dscribe.descriptors import SineMatrix, ACSF
from matminer.featurizers.site import LocalPropertyDifference
from matminer.featurizers.structure import (
    SiteStatsFingerprint,
    StructuralHeterogeneity,
)
import warnings

warnings.filterwarnings('ignore')


def trans_structure(cif, matrix):
    lattice = cif.lattice

    new_lattice_matrix = np.dot(matrix, lattice.matrix)
    new_lattice = Lattice(new_lattice_matrix)

    coords = cif.frac_coords
    rotated_coords = np.dot(coords, matrix.T)

    new_structure = Structure(
        new_lattice,
        cif.species,
        rotated_coords,
        coords_are_cartesian=False
    )

    return new_structure


def add_gaussian_noise(cif):
    lattice = cif.lattice
    a, b, c = lattice.a, lattice.b, lattice.c

    noise_ranges = {
        'x': (0.01 * a, 0.02 * a),
        'y': (0.01 * b, 0.02 * b),
        'z': (0.01 * c, 0.02 * c)
    }
    all_indices = list(range(len(cif)))
    num_to_select = int(0.25 * len(cif))
    selected_indices = random.sample(all_indices, num_to_select)

    for i in selected_indices:
        site = cif[i]
        frac_coords = site.frac_coords
        new_frac_coords = np.zeros(3)
        for dim in range(3):
            min_noise, max_noise = noise_ranges['xyz'[dim]]
            mu = 0
            sigma = (max_noise - min_noise) / 4
            cartesian_noise = np.random.normal(mu, sigma)
            lattice_param = [a, b, c][dim]
            frac_noise = cartesian_noise / lattice_param
            new_frac_coords[dim] = frac_coords[dim] + frac_noise
        new_frac_coords = np.mod(new_frac_coords, 1)
        cif[i] = site.species, new_frac_coords

    return cif


def cal_xrd(structure, two_theta_range=(10, 90)):
    c = XRDCalculator(wavelength='CuKa1')
    xrd_data = XRDCalculator.get_pattern(c, structure, scaled=True, two_theta_range=two_theta_range)
    return xrd_data


def spectrum(x, y, sigma, x_range, FWHM, a, omega):
    gE = []
    for xi in x_range:
        tot = 0
        for xj, o in zip(x, y):
            l = (FWHM / (2 * np.pi)) * (1 / ((xj - xi) ** 2 + 0.25 * FWHM ** 2))
            g = a * np.exp(-(xj - xi) ** 2 / (2 * sigma ** 2))
            p = omega * g + (1 - omega) * l
            tot += o * p
        gE.append(tot)
    return gE


def xrd(cif):
    x_min = 10
    x_max = 90
    x_num = int((x_max - x_min) / 0.02) + 1
    FWHM = 0.1
    sigma = FWHM * 0.42463
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    omega = 0.001
    x = np.linspace(x_min, x_max, num=x_num, endpoint=True)

    xrd_data = cal_xrd(cif, two_theta_range=(x_min, x_max))
    gxrd = spectrum(xrd_data.x, xrd_data.y, sigma, x, FWHM, a, omega)
    y = np.array(gxrd)
    y_max = y.max()
    y = y / y_max
    return y


def li_ACSF(cif):
    structure = AseAtomsAdaptor.get_atoms(cif)
    species = ['S', 'Sb', 'W', 'Ge', 'Fe', 'Er', 'Mg', 'La', 'Si', 'Hf', 'Cr', 'Eu', 'Sn', 'Se', 'Ga', 'Sr', 'Nd',
               'N', 'Cu', 'Sc', 'F', 'Ti', 'K', 'Co', 'Pr', 'Li', 'Mn', 'Yb', 'Zn', 'Nb', 'P', 'Al', 'C', 'In', 'V',
               'Bi', 'H', 'Gd', 'Mo', 'Br', 'O', 'Y', 'Sm', 'I', 'Na', 'Ta', 'Cl', 'Ba', 'Cd', 'Te', 'B', 'Ce', 'Zr']
    r_cut = 6.0
    g2_params = [[0.5, 0.0], [0.5, 2.0]]
    g4_params = [[0.005, 1, 1.0], [0.005, 1, -1.0]]

    acsf = ACSF(
        species=species,
        r_cut=r_cut,
        g2_params=g2_params,
        g4_params=g4_params
    )

    descriptors = acsf.create(structure)

    symbols = structure.get_chemical_symbols()
    li_indices = [j for j, sym in enumerate(symbols) if sym == "Li"]
    li_descriptors = descriptors[li_indices]
    return li_descriptors, li_indices


def extend_cell(cif, rmax=20):
    lengths = np.array(cif.lattice.lengths)
    a = rmax*2
    size = [1, 1, 1]
    for i in range(0, 3):
        size[i] = int((a + lengths[i] - 1) // lengths[i])
    return size


def get_distance_from_reference(cif):
    ref = cif.lattice.get_cartesian_coords([0.5, 0.5, 0.5])

    li_distances = {}
    for i, site in enumerate(cif):
        if site.specie.symbol == "Li":
            dist = np.linalg.norm(site.coords - np.array(ref))
            li_distances[i] = dist
    return li_distances


def mACSF(mycif, rmax=20, n_bins=5):
    cif = mycif.copy()
    vector = np.array(extend_cell(cif, rmax))
    cif = cif.make_supercell(vector)

    li_descriptors, li_indices = li_ACSF(cif)
    li_distances = get_distance_from_reference(cif)

    acsf_dim = li_descriptors.shape[-1]
    binned_acsf = np.zeros((n_bins, acsf_dim))
    counts = np.zeros(n_bins)

    bin_width = rmax / n_bins
    for i, li_idx in enumerate(li_indices):
        dist = li_distances[li_idx]
        bin_idx = int(dist // bin_width)
        if bin_idx < n_bins:
            binned_acsf[bin_idx] += li_descriptors[i]
            counts[bin_idx] += 1

    for i in range(n_bins):
        if counts[i] > 0:
            binned_acsf[i] /= counts[i]

    return binned_acsf.reshape(-1)


def mrdf(mycif, ngrid=400, rmax=20, sigma=0.2):
    cif = mycif.copy()
    vector = np.array(extend_cell(cif, rmax))
    supercell = cif.make_supercell(vector)
    li_distances = get_distance_from_reference(supercell)
    li_distances = np.array(list(li_distances.values()))

    bin_width = rmax / ngrid
    bins = np.linspace(0, rmax, ngrid + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    counts, _ = np.histogram(li_distances, bins=bins)
    shell_areas = 4 * np.pi * bin_centers ** 2 * bin_width
    shell_areas[shell_areas == 0] = bin_width ** 3
    total_li = len(li_distances)
    v_supercell = supercell.volume
    avg_density = total_li / v_supercell
    rdf = counts / (shell_areas * avg_density)
    sigma_index = sigma / bin_width
    rdf_smoothed = gaussian_filter1d(rdf, sigma_index)
    rdf_min = rdf_smoothed.min()
    rdf_max = rdf_smoothed.max()
    rdf_range = rdf_max - rdf_min
    rdf_normalized = (rdf_smoothed - rdf_min) / rdf_range
    return rdf_normalized


def extract_features_from_cif(structure):
    feature = {}
    li_indices = [j for j, site in enumerate(structure) if site.species_string == "Li+"]
    vnn = VoronoiNN(cutoff=13)
    channel_radii = []
    for idx in li_indices:
        env = vnn.get_voronoi_polyhedra(structure, idx)
        min_face_area = min([face["area"] for face in env.values()])
        channel_radii.append(np.sqrt(min_face_area / np.pi))
    feature["avg_channel_radius"] = np.mean(channel_radii)
    feature["min_channel_radius"] = np.min(channel_radii)

    lpd = LocalPropertyDifference()
    site_stats = SiteStatsFingerprint(lpd, stats=("mean", "std_dev"))
    site_stats_values = site_stats.featurize(structure)
    site_stats_labels = site_stats.feature_labels()
    feature.update(dict(zip(site_stats_labels, site_stats_values)))
    heterogeneity = StructuralHeterogeneity(stats=("mean", "avg_dev"))
    heterogeneity_values = heterogeneity.featurize(structure)
    heterogeneity_labels = heterogeneity.feature_labels()
    feature.update(dict(zip(heterogeneity_labels, heterogeneity_values)))
    feature["n_li"] = len(li_indices) / structure.volume

    return feature


def process(cif_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    sinematrix = SineMatrix(n_atoms_max=200, permutation="sorted_l2")
    cif_list = os.listdir(cif_path)
    i = 0
    for file in cif_list:
        i += 1
        name = file[:-4]
        print(" ")
        print("No.%d -- %s" % (i, name))

        parse = CifParser(cif_path + file)
        mycif = parse.get_structures(primitive=False)[0]
        if mycif.is_ordered:
            total_num = mycif.num_sites
            lengths = np.array(mycif.lattice.lengths)

            size = np.array([1, 1, 1])
            if total_num < 67:
                min_index = np.argmin(lengths)
                size[min_index] *= 3
            elif total_num < 100:
                min_index = np.argmin(lengths)
                size[min_index] *= 2

            output_path = os.path.join(out_path, name+".txt")
            row_names = ['RDF', 'XRD', 'SineMatrix', 'mACSF', 'Scalar']
            features = list(extract_features_from_cif(mycif).values())
            print("------step 1/5 finished------")
            myrdf = mrdf(mycif)
            print("------step 2/5 finished------")
            myxrd = xrd(mycif)
            print("------step 3/5 finished------")
            mysine = sinematrix.create(AseAtomsAdaptor.get_atoms(mycif))
            print("------step 4/5 finished------")
            myacsf = mACSF(mycif)
            print("------step 5/5 finished------")

            lists = [myrdf, myxrd, mysine, myacsf, features]

            with open(output_path, "w", encoding="utf-8") as f:
                for name, items in zip(row_names, lists):
                    line_content = " ".join(map(str, items))
                    line = f"{name}: {line_content}\n"
                    f.write(line)

        else:
            print("The crystal structure is disordered! The data processing progress will be skipped!")


if __name__ == "__main__":
    np.random.seed(224)
    random.seed(224)
    cif_folder_path = "to_your_cif_folder_path/"
    out_path = "output/"
    process(cif_path=cif_folder_path, out_path=out_path)
