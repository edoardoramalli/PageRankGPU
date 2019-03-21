import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy import sparse
import csv
from tqdm import trange, tqdm

DAMPING = 0.85

path_edge = "./pagerank_contest_edgelists/graph_small_e.edgelist"
path_vertex = "./pagerank_contest_edgelists/graph_small_v.edgelist"




def manage_vertex():
    dictionary = {}
    print("Creating dictionary...")
    with open(path_vertex, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='*')
        for row in csv_reader:
            name_site = (row[0].replace('"', '')).strip()
            index_site = (row[2].replace('"', '')).strip()
            dictionary[name_site] = index_site
    csv_file.close()
    print("Done!")

    return dictionary


dictionary = manage_vertex()
num_of_vertex = len(dictionary)
print(num_of_vertex)


def manage_edge():
    row = []
    column = []
    data = []
    print("Creating A matrix...")
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
    print("Done!")
    csv_file.close()

    

    data = np.array(data).astype(np.int32)
    row = np.array(row).astype(np.int32)
    column = np.array(column).astype(np.int32)

    a = num_of_vertex + 1 #fixes deep shit

    A = csr_matrix((data, (row,column)),(a, a))
    print("Shape A : ",A.get_shape())

    print("Creating d^-1 vector")
    d = []
    for row in A:
        elements = row.count_nonzero()
        d.append(1/elements if elements != 0 else 0)
    D_inv = csr_matrix(diags(d,0))
    print("Done!")

    print("Multiplying...")
    T = D_inv * A
    lil_t = lil_matrix(T)
    print("Done!")

    uniform = (1/a) * np.ones(a)

    # for i in trange(a):
    #     if lil_t[i].count_nonzero() == 0:
    #         lil_t[i] = uniform
    
    T = csr_matrix(lil_t)

    print("Transposing T...")
    T_trans = T.transpose()
    T_trans.data = DAMPING*T_trans.data + (1-DAMPING)/a
    print("Done!")
    return T_trans

def compute_damping_matrix ():
    # (1-d)|E|/|V|
    a = (1 - DAMPING) * (1/num_of_vertex)
    return a

damping_matrix = compute_damping_matrix()

Tt = manage_edge()
Tt = DAMPING * Tt


with open("data.csv", "w") as f: 
    writer = csv.writer(f,delimiter = ',')
    writer.writerow([len(Tt.indptr)])
    writer.writerow(Tt.indptr)
    writer.writerow([len(Tt.indices)])
    writer.writerow(Tt.indices)
    writer.writerow([len(Tt.data)])
    writer.writerow(Tt.data)
    writer.writerow([damping_matrix])
f.close()


# if __name__ == "__main__":
#     #TODO add command line parameters
#     # parser = argparse.ArgumentParser(description='Prepare dataset matrices')
#     # parser.add_argument()
#     pass


