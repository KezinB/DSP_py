k = int(input("Enter the number of vectors (k): "))
input_vectors = []
for i in range(k):
    vector = [float(x) for x in input(f"Enter vector {i + 1} as space-separated values: ").split()]
    input_vectors.append(vector)
    print(f"vector{i+1}:{vector}")
orthogonal_basis = []
orthonormal_sets = []
for vector in input_vectors:
    for basis_vector in orthogonal_basis:
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector, basis_vector))
        vector = [v - dot_product / sum(vi**2 for vi in basis_vector) * bi for v,
bi in zip(vector, basis_vector)]
orthogonal_basis.append(vector)
for basis_vector in orthogonal_basis:
    magnitude = sum(x**2 for x in basis_vector)**0.5
    orthonormal_sets.append([x / magnitude for x in basis_vector])
print(f"The orthogonal bases are {{a1, a2, a3}}: {orthogonal_basis}")
print(f"The orthonormal sets are {{d1, d2, d3}}: {orthonormal_sets}")
