import numpy as np
import math
import csv

SOGLIA = 0.000001
DAMPING = 0.85

filename = "data_small.csv"

with open(filename) as infile:
    r = csv.reader(infile)
    len_ptr = next(r)[0]
    ptr = list(map(int, next(r)))
    len_index = next(r)[0]
    index = list(map(int, next(r)))
    len_data = next(r)[0]
    data = list(map(float, next(r)))
    damping_matrix = next(r)[0]
    len_empty = next(r)[0]
    empty = list(map(int, next(r)))

number_of_vertex = int(len_ptr) - 1

prima = damping_matrix

pk = np.dot(np.ones(number_of_vertex), 1 / number_of_vertex)


def compute_row(v):
    rr = []
    for i in range(0, len(v) - 1):
        old = i
        new = i + 1
        if v[old] != v[new]:
            for j in range(0, v[new] - v[old]):
                rr.append(i)
    return rr


rows = compute_row(ptr)


def check_norm(v1, v2):
    summ = 0
    for i in range(0, len(v1)):
        tmp = float(v1[i]) - float(v2[i])
        summ = summ + (tmp ** 2)
    return math.sqrt(summ)


def iterate(d, column, pointer, init, damp_matrix, row, e):
    termina = True
    old = init
    count = 0

    while termina:
        empty_contrib = 0
        for i in range(0, len(e)):
            empty_contrib = empty_contrib + (float(old[e[i]]) * (1 / (len(ptr) - 1)) * DAMPING)

        tmp = float(damp_matrix) + float(empty_contrib)
        new = [tmp] * (len(pointer) - 1)

        for i in range(0, len(row)):
            new[row[i]] = float(new[row[i]]) + float(old[column[i]]) * float(d[i])

        errore = check_norm(old, new)
        if (errore - SOGLIA) < SOGLIA:
            termina = False
        print("----- ITERAZIONE ", count, " -----")
        print("ERRORE : ", errore)
        # print("SUM PK", round(sum(list(map(float, new))), 5))
        old = new
        count = count + 1
        if count == 10:
            with open("pk_" + filename, "w+") as outfile:
                for it in new:
                    outfile.write(str(it))
                    outfile.write("\n")
            break
    # print("PK : ", new)


iterate(data, index, ptr, pk, prima, rows, empty)
