from scipy.sparse import csr_matrix, diags, lil_matrix
import csv
import sys
import getopt
import time
from tqdm import tqdm


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
    destination_path = ""
    damping_factor = 0.85
    try:
        opts, args = getopt.getopt(argv, "sfv:e:o:d:", ["efile=", "vfile=", "ofile=", "dfile="])
    except getopt.GetoptError:
        print(Bcolors.FAIL + "Syntax Error " + str(getopt.GetoptError) + Bcolors.ENDC)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            print(Bcolors.OKGREEN + "Default Path Full Selected" + Bcolors.ENDC)
            path_edge = "./pagerank_contest_edgelists/graph_full_e.edgelist"
            path_vertex = "./pagerank_contest_edgelists/graph_full_v.edgelist"
            destination_path = "data_full.csv"
        elif opt == '-s':
            print(Bcolors.OKGREEN + "Default Path Small Selected" + Bcolors.ENDC)
            path_edge = "./pagerank_contest_edgelists/graph_small_e.edgelist"
            path_vertex = "./pagerank_contest_edgelists/graph_small_v.edgelist"
            destination_path = "data_small.csv"
        elif opt in ("-v", "--vfile"):
            path_vertex = arg
        elif opt in ("-e", "--efile"):
            path_edge = arg
        elif opt in ("-o", "--ofile"):
            destination_path = arg
        elif opt in ("-d", "--dfile"):
            damping_factor = arg
    if path_edge == "":
        print(Bcolors.FAIL + "Missing Path Edge" + Bcolors.ENDC)
        exit(2)
    if path_vertex == "":
        print(Bcolors.FAIL + "Missing Path Vertex" + Bcolors.ENDC)
        exit(2)
    if destination_path == "":
        print(Bcolors.FAIL + "Missing Path Destination" + Bcolors.ENDC)
        exit(2)
    if damping_factor == "":
        print(Bcolors.FAIL + "Invalid Damping" + Bcolors.ENDC)
        exit(2)
    damping_factor = float(damping_factor)
    if not 0 < damping_factor < 1:
        print(Bcolors.FAIL + "Invalid Damping (0 < d < 1)" + Bcolors.ENDC)
        exit(2)

    print(Bcolors.WARNING + "Edge Path : " + path_edge + Bcolors.ENDC)
    print(Bcolors.WARNING + "Vertex Path : " + path_vertex + Bcolors.ENDC)
    print(Bcolors.WARNING + "Output Path : " + destination_path + Bcolors.ENDC)
    print(Bcolors.WARNING + "Damping Factor : " + str(damping_factor) + Bcolors.ENDC)

    return path_edge, path_vertex, destination_path, damping_factor


def compute_empty_row(vector):
    result = []
    for i in range(0, len(vector) - 1):
        if vector[i] == vector[i + 1]:
            result.append(i)
    return result


def manage_vertex(path_vertex):
    dictionary = {}
    print(Bcolors.OKBLUE + "Creating Dictionary..." + Bcolors.ENDC)
    with open(path_vertex, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='*')
        it = 0
        for row in tqdm(csv_reader):
            name_site = (row[0].replace('"', '')).strip()
            dictionary[name_site] = it
            it = it + 1
    csv_file.close()
    return dictionary, int(it)


def manage_edge(path_edge, dictionary, num_of_vertex):
    row = []
    column = []
    data = []
    print(Bcolors.OKBLUE + "Creating A matrix..." + Bcolors.ENDC)
    i = 0
    with open(path_edge, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for line in tqdm(csv_reader):
            name_site = (line[0].replace('"', '')).strip()
            reported_site = (line[1].replace('"', '')).strip()
            source_index = (dictionary[name_site])
            destination_index = (dictionary[reported_site])
            if source_index == None:
                print(Bcolors.FAIL + "Missing Source Index : " + name_site + Bcolors.ENDC)
            if destination_index == None:
                print(Bcolors.FAIL + "Missing Destination Index : " + reported_site + Bcolors.ENDC)
            if source_index != None and destination_index != None:
                row.append(source_index)
                column.append(destination_index)
                data.append(1)
                i = i + 1

    print(Bcolors.OKGREEN + "Number of Edge : " + str(i) + Bcolors.ENDC)
    csv_file.close()

    dimension = int(num_of_vertex)

    a_matrix = csr_matrix((data, (row, column)), (dimension, dimension))

    print(Bcolors.OKBLUE + "Creating d^-1 vector..." + Bcolors.ENDC)
    d = a_matrix.sum(axis=1)
    d_pr = []
    for el in tqdm(d.transpose().tolist()[0]):
        d_pr.append((1.0 / el) if el != 0 else 0)

    d_inv = diags(d_pr, 0)

    print(Bcolors.OKBLUE + "Multiplying d_inv * a_matrix..." + Bcolors.ENDC)
    t_matrix = d_inv.dot(a_matrix)

    print(Bcolors.OKBLUE + "Transposing T..." + Bcolors.ENDC)
    t_trans = csr_matrix(t_matrix.transpose())

    return t_trans, t_matrix


def compute_damping_matrix(num_of_vertex, damping_factor):
    a = (1 - damping_factor) / num_of_vertex
    return a


def compute_row(v):
    rr = []
    for i in range(0, len(v) - 1):
        old = i
        new = i + 1
        if v[old] != v[new]:
            for j in range(0, v[new] - v[old]):
                rr.append(i)
    return rr


def manage_time(rr):
    if rr <= 60:
        t = round(rr, 2)
        return "Elapsed Time : " + str(t) + " sec."
    elif rr < 60 * 60:
        t = round(rr / 60, 2)
        return "Elapsed Time : " + str(t) + " min."
    else:
        t = round(rr / (60 * 60), 2)
        return "Elapsed Time : " + str(t) + " h."


def main(argv):
    path_edge, path_vertex, destination_path, damping_factor = parse_input(argv)

    dictionary, num_of_vertex = manage_vertex(path_vertex)

    print(Bcolors.OKGREEN + "Number of Vertex : " + str(num_of_vertex) + Bcolors.ENDC)

    t, t_before_trans = manage_edge(path_edge, dictionary, num_of_vertex)

    print(Bcolors.OKBLUE + "Compute Damping Matrix (Single Value)..." + Bcolors.ENDC)

    damping_matrix = compute_damping_matrix(num_of_vertex, damping_factor)

    print(Bcolors.OKBLUE + "Multiply T * Damping..." + Bcolors.ENDC)
    t_final = damping_factor * t

    print(Bcolors.OKBLUE + "Compute Empty Row..." + Bcolors.ENDC)
    empty_row = compute_empty_row(t_before_trans.indptr)
    lunghezza_empty_row = len(empty_row)

    print(Bcolors.OKBLUE + "Compute Row with Data..." + Bcolors.ENDC)
    rows = compute_row(t_final.indptr)

    print(Bcolors.OKGREEN + "Write CSV" + Bcolors.ENDC)

    with open(destination_path, "w+") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([len(t_final.indptr) - 1])
        writer.writerow([len(t_final.indices)])
        writer.writerow(rows)
        writer.writerow(t_final.indices)
        writer.writerow(t_final.data)
        writer.writerow([damping_matrix])
        writer.writerow([lunghezza_empty_row])
        writer.writerow(empty_row)
    f.close()


if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print(Bcolors.HEADER + manage_time(end - start) + Bcolors.ENDC)
