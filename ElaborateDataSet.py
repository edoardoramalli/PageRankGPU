path_edge = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_e.edgelist"
path_vertex = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_v.edgelist"

import csv

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
    #     writer.writerows(dictionary)
    #
    # csvFile.close()

    return dictionary




dictionary = manage_vertex()
num_of_vertex = len(dictionary)

matrix = [[0]]
for i in range(1, num_of_vertex):
    matrix.append([i])


def manage_edge():
    with open(path_edge) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            reported_site = (row[1].replace('"', '')).strip()
            index_of_name_site = int(dictionary.get(name_site))
            index_of_reported_site = int(dictionary.get(reported_site))

            matrix[index_of_name_site].append(index_of_reported_site)

    with open("matrixA.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)

    d = []
    for i in range (0,len(matrix)):
        d.append(len(matrix[i])-1)

    d_inv =[]
    for i in range (0,len(d)):
        if d[i] != 0:
            d_inv.append(round(1/d[i],6))
        else:
            d_inv.append(0)

    with open("matrixD.csv", "w") as f:
        writer = csv.writer(f, delimiter='#')
        writer.writerow(d)

    with open("matrixDinv.csv", "w") as f:
        writer = csv.writer(f, delimiter='#')
        writer.writerow(d_inv)


manage_edge()
