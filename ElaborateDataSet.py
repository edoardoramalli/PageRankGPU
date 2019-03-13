
import csv
from scipy.sparse import csr_matrix

path_edge = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_e.edgelist"
path_vertex = "/users/edoardo/downloads/pagerank_contest_edgelists/graph_small_v.edgelist"




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

matrix_A = [[0]]

# matrix_A = [([i] for i in range(0, num_of_vertex) )]
for i in range(1, num_of_vertex):
    matrix_A.append([i])


def manage_edge():
    with open(path_edge) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            reported_site = (row[1].replace('"', '')).strip()
            index_of_name_site = int(dictionary.get(name_site))
            index_of_reported_site = int(dictionary.get(reported_site))

            matrix_A[index_of_name_site].append(index_of_reported_site)



    with open("matrix_AA.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([num_of_vertex])
        writer.writerows(matrix_A)

    d = []
    for i in range(0, len(matrix_A)):
        d.append(len(matrix_A[i]) - 1)

    d_inv = []
    for i in range(0, len(d)):
        if d[i] != 0:
            d_inv.append(round(1 / d[i], 6))
        else:
            d_inv.append(0)

    # with open("matrix_AD.csv", "w") as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(d)
    #
    # with open("matrix_ADinv.csv", "w") as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(d_inv)

    return d_inv


d_inv = manage_edge()


def computeT():
    equiprobability = round(1 / num_of_vertex, 6)
    m = csr_matrix((number_of_vertex, number_of_vertex), dtype=np.int32)
    print ("equiporbo " + str(equiprobability))
    for i in range(0, len(matrix_A)):
        if i == 3:
            if d_inv[i] == 0:
                lista = [equiprobability,"#"]
                print ("ESEMPIOppppppppppp")
            else:
                print(matrix_A[i].pop(0))
                lista = d_inv[i] + matrix_A[i].pop(0)
            matrix_A[i] = lista

    for i in range (0,10):
        print(matrix_A[i])

    # with open("matrix_T.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([num_of_vertex])
    #     writer.writerows(matrix_A)


computeT()

# if __name__ == "__main__":
#     #TODO add command line parameters
#     # parser = argparse.ArgumentParser(description='Prepare dataset matrices')
#     # parser.add_argument()
#     pass
