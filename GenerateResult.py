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
    path_vertex = ""
    destination_path = ""
    pk_path = ""
    try:
        opts, args = getopt.getopt(argv, "sfv:p:o:", ["vfile=", "pfile=", "ofile="])
    except getopt.GetoptError:
        print(Bcolors.FAIL + "Syntax Error" + Bcolors.ENDC)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            print(Bcolors.OKGREEN + "Default Path Full Selected" + Bcolors.ENDC)
            path_vertex = "./pagerank_contest_edgelists/graph_full_v.edgelist"
            destination_path = "result_data_full.csv"
            pk_path = "pk_data_full.csv"
        elif opt == '-s':
            print(Bcolors.OKGREEN + "Default Path Small Selected" + Bcolors.ENDC)
            path_vertex = "./pagerank_contest_edgelists/graph_small_v.edgelist"
            destination_path = "result_data_small.csv"
            pk_path = "pk_data_small.csv"
        elif opt in ("-v", "--vfile"):
            path_vertex = arg
        elif opt in ("-o", "--ofile"):
            destination_path = arg
        elif opt in ("-p", "--pfile"):
            pk_path = arg
    if pk_path == "":
        print(Bcolors.FAIL + "Missing Path PageRank" + Bcolors.ENDC)
        exit(2)
    if path_vertex == "":
        print(Bcolors.FAIL + "Missing Path Vertex" + Bcolors.ENDC)
        exit(2)
    if destination_path == "":
        print(Bcolors.FAIL + "Missing Path Destination" + Bcolors.ENDC)
        exit(2)
    return pk_path, path_vertex, destination_path


def manage_vertex(path):
    vertex = []
    with open(path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='*')
        for row in csv_reader:
            name_site = row[0].strip()
            vertex.append('"' + name_site + '"')
    csv_file.close()
    return vertex


def manage_pk(path):
    pk = []
    with open(path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            value = (row[0].replace('"', '')).strip()
            pk.append(value)
    csv_file.close()
    return pk


def main(argv):
    pk_path, path_vertex, destination_path = parse_input(argv)

    list_of_vertex = manage_vertex(path_vertex)
    print(Bcolors.OKBLUE + "Reading Vertex List" + Bcolors.ENDC)

    pk_result = manage_pk(pk_path)
    print(Bcolors.OKBLUE + "Reading PageRank Result" + Bcolors.ENDC)

    final_list = []

    if len(list_of_vertex) != len(pk_result):
        print(Bcolors.FAIL + "Different Dimension PK-VERTEX" + Bcolors.ENDC)
        exit(2)

    print(Bcolors.OKBLUE + "Creating Final Result" + Bcolors.ENDC)
    for i in range(0, len(list_of_vertex)):
        final_list.append(list_of_vertex[i] + " " + pk_result[i])

    print(Bcolors.OKGREEN + "Writing CSV" + Bcolors.ENDC)
    with open(destination_path, "w+") as outfile:
        for it in tqdm(final_list):
            outfile.write(str(it))
            outfile.write("\n")


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
