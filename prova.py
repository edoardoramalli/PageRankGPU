from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

m = np.array(([0, 0, 1], [1, 0, 1], [0, 0, 0]))


a = csr_matrix(m)

print(a.toarray())

print("Data ", a.data)
print("Ptr ", a.indptr)
print("Column ", a.indices)

b = csr_matrix(a.transpose())

print("BBBBBBBB")
print(b.toarray())

print("Data ", b.data)
print("Ptr ", b.indptr)
print("Column ", b.indices)
