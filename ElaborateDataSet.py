path_edge = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_e.edgelist"
path_vertex = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_v.edgelist"

import csv
import numpy as np


def manage_vertex():
    dictionary = {}
    number_of_vertex = 0
    with open(path_vertex) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='*')
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            index_site = (row[2].replace('"', '')).strip()
            dictionary[name_site] = index_site
            number_of_vertex = number_of_vertex + 1

    # with open('vertex.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(lista)
    #
    # csvFile.close()

    return dictionary


dictionary = manage_vertex()
num_of_vertex = len(dictionary)

matrix = {}
for i in range(0, num_of_vertex):
    matrix[i] = []



def manage_edge():
    with open(path_edge) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        count = 0
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            reported_site = (row[1].replace('"', '')).strip()
            index_of_name_site = int(dictionary.get(name_site))
            index_of_reported_site = int(dictionary.get(reported_site))

            lista = matrix[index_of_name_site]

            if lista != []:
                print("evviva")
                print(type(lista))
                print(lista)
            lista = lista.append(index_of_reported_site)
            matrix[index_of_name_site] = lista


    with open('dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in matrix.items():
            writer.writerow([key, value])






manage_edge()

print(type(matrix[87345]))
