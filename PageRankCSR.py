import numpy as np
import csv
import sys
import getopt
import time


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
    filename = ""
    destination_path = ""
    soglia = 1
    damping = 0
    try:
        opts, args = getopt.getopt(argv, "sfi:o:d:t:", ["dfile=", "ifile=", "ofile=", "tfile=", "dfile="])
    except getopt.GetoptError:
        print(Bcolors.FAIL + "Syntax Error" + Bcolors.ENDC)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            print(Bcolors.OKGREEN + "Default Path Full Selected" + Bcolors.ENDC)
            filename = "data_full.csv"
            destination_path = "pk_" + filename
            soglia = 0.000001
            damping = 0.85
        elif opt == '-s':
            print(Bcolors.OKGREEN + "Default Path Small Selected" + Bcolors.ENDC)
            filename = "data_small.csv"
            destination_path = "pk_" + filename
            soglia = 0.000001
            damping = 0.85
        elif opt in ("-i", "--ifile"):
            filename = arg
            destination_path = "pk_" + filename
            soglia = 0.000001
            damping = 0.85
        elif opt in ("-o", "--efile"):
            destination_path = arg
        elif opt in ("-t", "--ofile"):
            soglia = arg
        elif opt in ("-d", "--dfile"):
            damping = arg
    if filename == "":
        print(Bcolors.FAIL + "Missing Path Input" + Bcolors.ENDC)
        exit(2)
    if destination_path == "":
        print(Bcolors.FAIL + "Missing Name Output" + Bcolors.ENDC)
        exit(2)
    if not 0 <= soglia < 1:
        print(Bcolors.FAIL + "Invalid Threshold" + Bcolors.ENDC)
        exit(2)
    if damping == 0:
        print(Bcolors.FAIL + "Invalid Damping" + Bcolors.ENDC)
        exit(2)
    return filename, destination_path, soglia, damping


def check_norm(v1, v2):
    return np.linalg.norm(np.array(v2) - np.array(v1), 2)


def iterate(d, column, num_of_vertex, init, damp_matrix, row, e, soglia, damping, destination):
    termina = True
    old = init
    count = 0

    while termina:
        empty_contrib = 0
        for i in range(0, len(e)):
            empty_contrib = empty_contrib + (float(old[e[i]]) * (1 / num_of_vertex) * damping)

        tmp = float(damp_matrix) + float(empty_contrib)
        new = [tmp] * num_of_vertex

        for i in range(0, len(row)):
            new[row[i]] = float(new[row[i]]) + float(old[column[i]]) * float(d[i])

        errore = check_norm(old, new)
        if errore <= soglia:
            termina = False
        print("----- ITERAZIONE ", count, " -----")
        print("ERRORE : ", errore)
        # print("SUM PK", round(sum(list(map(float, new))), 5))
        old = new
        count = count + 1

    print(Bcolors.OKGREEN + "Write CSV" + Bcolors.ENDC)

    with open(destination, "w+") as outfile:
        for it in new:
            outfile.write(str(it))
            outfile.write("\n")


def main(argv):
    filename, destination_path, soglia, damping = parse_input(argv)
    print(Bcolors.HEADER + "Damping : " + str(damping) + " Threshold : " + str(soglia) + Bcolors.ENDC)

    with open(filename) as infile:
        r = csv.reader(infile)
        len_ptr = next(r)[0]
        len_index = next(r)[0]
        rows = list(map(int, next(r)))
        index = list(map(int, next(r)))
        data = list(map(float, next(r)))
        damping_matrix = next(r)[0]
        len_empty = next(r)[0]
        empty = list(map(int, next(r)))

    number_of_vertex = int(len_ptr)

    prima = float(damping_matrix)

    pk = np.dot(np.ones(number_of_vertex), 1 / number_of_vertex)

    iterate(data, index, number_of_vertex, pk, prima, rows, empty, soglia, float(damping), destination_path)


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


if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print(Bcolors.HEADER + manage_time(end - start) + Bcolors.ENDC)
