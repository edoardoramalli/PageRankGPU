import numpy as np
import math

DAMPING = 0.85 # <-------------------------------GUARDAAAAA
SOGLIA = 0.000001

a = np.array(([0,1,0,1],
              [0,0,1,1],
              [0,0,0,0],
              [1,0,0,0]))

# a = np.array(([0, 1, 0],
#               [1, 0, 1],
#               [0, 1, 0]))

# a = np.array(([0, 1, 0, 0, 0, 0],
#               [0, 0, 1, 1, 1, 0],
#               [0, 1, 0, 1, 1, 0],
#               [0, 1, 1, 0, 1, 0],
#               [0, 1, 1, 1, 0, 1],
#               [0, 0, 0, 0, 0, 0]))

number_of_vertex = int(a.shape[0])

sum_vector = []
for i in range (0, number_of_vertex):
    sum_vector.append(sum(a[i]))

sum_inv = []
for i in range (0, number_of_vertex):
    if sum_vector[i] == 0:
        sum_inv.append(0)
    else:
        sum_inv.append(1/sum_vector[i])

diag = np.diag(sum_inv)


t = np.dot(diag, a)

for i in range(0, number_of_vertex):
    somma = sum(t[i])
    if somma == 0:
        uni = np.dot(np.ones(number_of_vertex), 1/number_of_vertex)
        t[i] = uni

t_trans = t.transpose()
t_trans_damp = t_trans * DAMPING

seconda = t_trans_damp

prima = np.dot(np.ones((number_of_vertex,number_of_vertex)),(1-DAMPING)/number_of_vertex)

pk = np.dot(np.ones(number_of_vertex),1/number_of_vertex)


def check_norm(v1, v2):
    sum = 0
    for i in range (0,len(v1)):
        tmp = v1[i] - v2[i]
        sum = sum + (tmp ** 2)
    return math.sqrt(sum)




def iterate(p, s, init):
    termina = True
    old = init
    count = 0
    parz = p + s

    while termina:
        new = []
        new = np.dot(parz, old)
        errore = check_norm(old,new)
        if (errore-SOGLIA) < SOGLIA:
            termina = False
        print("----- ITERAZIONE ", count, " -----")
        print("PK : ", new)
        print("ERRORE : ", errore)
        print("SUM PK", round(sum(new),5))
        old = new
        count = count + 1

iterate(prima,seconda,pk)
