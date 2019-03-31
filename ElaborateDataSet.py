import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
import csv
import sys
import getopt

from tqdm import tqdm

DAMPING = 0.85


# python3 -e ./pagerank_contest_edgelists/graph_full_e.edgelist -v ./pagerank_contest_edgelists/graph_full_v.edgelist


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_input(argv):
    path_edge = ""
    path_vertex = ""
    try:
        opts, args = getopt.getopt(argv, "sfv:e:", ["efile=", "vfile="])
    except getopt.GetoptError:
        print(Bcolors.FAIL + "Syntax Error" + Bcolors.ENDC)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            print(Bcolors.OKGREEN + "Default Path Full Selected" + Bcolors.ENDC)
            path_edge = "./pagerank_contest_edgelists/graph_full_e.edgelist"
            path_vertex = "./pagerank_contest_edgelists/graph_full_v.edgelist"
        elif opt == '-s':
            print(Bcolors.OKGREEN + "Default Path Small Selected" + Bcolors.ENDC)
            path_edge = "./pagerank_contest_edgelists/graph_small_e.edgelist"
            path_vertex = "./pagerank_contest_edgelists/graph_small_v.edgelist"
        elif opt in ("-v", "--vfile"):
            path_vertex = arg
        elif opt in ("-e", "--efile"):
            path_edge = arg
    if (path_edge == "") or (path_vertex == ""):
        print(Bcolors.FAIL + "Missing Path" + Bcolors.ENDC)
        exit(2)
    return path_edge, path_vertex


def manage_vertex(path_vertex):
    dictionary = {}
    print(Bcolors.OKBLUE + "Creating dictionary..." + Bcolors.ENDC)
    with open(path_vertex, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='*')
        tmp = -1
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            index_site = (row[2].replace('"', '')).strip()
            dictionary[name_site] = index_site
            tmp = index_site
    csv_file.close()
    return dictionary, int(tmp)


def manage_edge(path_edge, dictionary, num_of_vertex):
    row = []
    column = []
    data = []
    print(Bcolors.OKBLUE + "Creating A matrix..." + Bcolors.ENDC)
    i = 0
    with open(path_edge, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for line in csv_reader:
            name_site = (line[0].replace('"', '')).strip()
            reported_site = (line[1].replace('"', '')).strip()
            source_index = (dictionary.get(name_site))
            destination_index = (dictionary.get(reported_site))
            row.append(source_index)
            column.append(destination_index)
            data.append(1)
            i = i + 1
    print(Bcolors.OKGREEN + "Num of edge " + str(i) + Bcolors.ENDC)
    csv_file.close()

    data = np.array(data).astype(np.int32)
    row = np.array(row).astype(np.int32)
    column = np.array(column).astype(np.int32)

    dimension = int(num_of_vertex) + 1  # fixes deep

    a_matrix = csr_matrix((data, (row, column)), (dimension, dimension))

    print(Bcolors.OKBLUE + "Creating d^-1 vector" + Bcolors.ENDC)
    d = []
    for row in tqdm(a_matrix):
        elements = row.count_nonzero()
        d.append(1 / elements if elements != 0 else 0)
    d_inv = csr_matrix(diags(d, 0))

    print(Bcolors.OKBLUE + "Multiplying d_inv * a_matrix." + Bcolors.ENDC)
    t_matrix = d_inv * a_matrix
    lil_t = lil_matrix(t_matrix)

    t_matrix = csr_matrix(lil_t)

    print(Bcolors.OKBLUE + "Transposing T..." + Bcolors.ENDC)
    t_trans = t_matrix.transpose()
    t_trans.data = DAMPING * t_trans.data + (1 - DAMPING) / dimension
    return t_trans


def compute_damping_matrix(num_of_vertex):
    # (1-d)|E|/|V|
    a = (1 - DAMPING) * (1 / int(num_of_vertex))
    return a


def compute_empty_row(vector):
    old = 0
    new = 1
    result = []
    for i in range(0, len(vector) - 1):
        if vector[old] == vector[new]:
            result.append(old)
        old = old + 1
        new = new + 1
    return result


def main(argv):
    path_edge, path_vertex = parse_input(argv)

    dictionary, num_of_vertex = manage_vertex(path_vertex)

    print(Bcolors.OKGREEN + "Number of Vertex " + str(num_of_vertex) + Bcolors.ENDC)

    t = manage_edge(path_edge, dictionary, num_of_vertex)

    print(Bcolors.OKBLUE + "Compute Damping Matrix (Single Value)" + Bcolors.ENDC)

    damping_matrix = compute_damping_matrix(num_of_vertex)

    print(Bcolors.OKBLUE + "Multiply T * Damping" + Bcolors.ENDC)
    t = DAMPING * t

    print(Bcolors.OKBLUE + "Compute Empty Row" + Bcolors.ENDC)
    empty_row = compute_empty_row(t.indptr)
    lunghezza = len(empty_row)

    print(Bcolors.OKBLUE + "Write CSV" + Bcolors.ENDC)

    with open("data.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([len(t.indptr)])
        writer.writerow(t.indptr)
        writer.writerow([len(t.indices)])
        writer.writerow(t.indices)
        writer.writerow([len(t.data)])
        writer.writerow(t.data)
        writer.writerow([damping_matrix])
        writer.writerow([lunghezza])
        writer.writerow(empty_row)
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
