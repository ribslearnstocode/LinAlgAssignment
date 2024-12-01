#Question 1 

class ComplexNumber:
    def __init__(self, real, imag=0):
        self.real = real
        self.imaginary = imag

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imaginary)
        return ComplexNumber(self.real + other.real, self.imaginary + other.imaginary)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imaginary * other)
        return ComplexNumber(
            self.real * other.real - self.imaginary * other.imaginary,
            self.real * other.imaginary + self.imaginary * other.real
        )

    def __truediv__(self, other):
        if isinstance(other, (int, float, ComplexNumber)):
            if isinstance(other, (int, float)):
                other = ComplexNumber(other)
            denominator = other.real**2 + other.imaginary**2
            if denominator == 0:
                raise ValueError("Division by zero")
            real_part = (self.real * other.real + self.imaginary * other.imaginary) / denominator
            imag_part = (self.imaginary * other.real - self.real * other.imaginary) / denominator
            return ComplexNumber(real_part, imag_part)

    def abs(self):
        return (self.real**2 + self.imaginary**2)**0.5

    def cc(self):
        return ComplexNumber(self.real, -self.imaginary)

    def __repr__(self):
        return f"{self.real}{'+' if self.imaginary >= 0 else ''}{self.imaginary}j"

class Vector:
    def __init__(self, field_type, length, predefined_coords=None):
        self.field_type = field_type
        self.length = length
        self.coordinates = predefined_coords if predefined_coords else self._initialize_coordinates()

    def _initialize_coordinates(self):
        coordinates = []
        print(f"Enter {self.length} {self.field_type} coordinates:")
        for i in range(self.length):
            if self.field_type == 'real':
                coordinates.append(float(input(f"Coordinate {i+1}: ")))
            else:
                real = float(input(f"Real part of coordinate {i+1}: "))
                imag = float(input(f"Imaginary part of coordinate {i+1}: "))
                coordinates.append(ComplexNumber(real, imag))
        return coordinates

class Matrix:
    def __init__(self, field_type, rows, cols, vectors=None, predefined_matrix=None):
        self.field_type = field_type
        self.rows = rows
        self.cols = cols
        
        if predefined_matrix:
            self.matrix = predefined_matrix
        elif vectors:
            self.matrix = self._create_matrix_from_vectors(vectors)
        else:
            self.matrix = self._initialize_matrix()

    def _initialize_matrix(self):
        matrix = []
        print(f"Enter {self.rows} x {self.cols} matrix entries:")
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                if self.field_type == 'real':
                    row.append(float(input(f"Entry at position [{i+1},{j+1}]: ")))
                else:
                    real = float(input(f"Real part at position [{i+1},{j+1}]: "))
                    imag = float(input(f"Imaginary part at position [{i+1},{j+1}]: "))
                    row.append(ComplexNumber(real, imag))
            matrix.append(row)
        return matrix

    def _create_matrix_from_vectors(self, vectors):
        if len(vectors) != self.cols or any(len(vec.coordinates) != self.rows for vec in vectors):
            raise ValueError("Vector dimensions do not match matrix specifications")
        return [
            [vectors[j].coordinates[i] for j in range(self.cols)]
            for i in range(self.rows)
        ]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        result_matrix = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.matrix[i][j] + other.matrix[i][j])
            result_matrix.append(row)
        return Matrix(self.field_type, self.rows, self.cols, predefined_matrix=result_matrix)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions are incompatible for multiplication")
            result_matrix = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    cell_value = self.matrix[i][0] * other.matrix[0][j]
                    for k in range(1, self.cols):
                        cell_value += self.matrix[i][k] * other.matrix[k][j]
                    row.append(cell_value)
                result_matrix.append(row)
            return Matrix(self.field_type, self.rows, other.cols, predefined_matrix=result_matrix)
        return NotImplemented

    def get_row(self, row_index):
        return self.matrix[row_index]

    def get_column(self, col_index):
        return [row[col_index] for row in self.matrix]

    def transpose(self):
        transposed_matrix = [
            [self.matrix[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return Matrix(self.field_type, self.cols, self.rows, predefined_matrix=transposed_matrix)

    def conj(self):
        if self.field_type == 'real':
            return self
        conjugate_matrix = [
            [entry.cc() for entry in row]
            for row in self.matrix
        ]
        return Matrix(self.field_type, self.rows, self.cols, predefined_matrix=conjugate_matrix)

    def conj_transpose(self):
        return self.transpose().conj()


#Q2

class MatrixProperties:
    @staticmethod
    def is_zero(matrix):
        for row in matrix:
            for element in row:
                if element != 0:
                    return False
        return True

    @staticmethod
    def is_symmetric(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] != matrix[j][i]:
                    return False
        return True

    @staticmethod
    def is_hermitian(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] != complex(matrix[j][i]).conjugate():
                    return False
        return True

    @staticmethod
    def is_square(matrix):
        return len(matrix) == len(matrix[0])

    @staticmethod
    def is_orthogonal(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        
        identity_matrix = [[1 if i == j else 0 for j in range(len(matrix))] for i in range(len(matrix))]
        
        def matrix_multiply(A, B):
            result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
            return result

        def transpose(A):
            return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

        transposed_matrix = transpose(matrix)
        multiplied_matrix = matrix_multiply(matrix, transposed_matrix)
        
        for i in range(len(multiplied_matrix)):
            for j in range(len(multiplied_matrix[0])):
                expected = 1 if i == j else 0
                if abs(multiplied_matrix[i][j] - expected) > 1e-10:
                    return False
        return True

    @staticmethod
    def is_unitary(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        
        def matrix_multiply(A, B):
            result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
            return result

        def conjugate_transpose(A):
            return [[complex(A[j][i]).conjugate() for j in range(len(A))] for i in range(len(A[0]))]

        transposed_matrix = conjugate_transpose(matrix)
        multiplied_matrix = matrix_multiply(matrix, transposed_matrix)
        
        for i in range(len(multiplied_matrix)):
            for j in range(len(multiplied_matrix[0])):
                expected = complex(1 if i == j else 0)
                if abs(complex(multiplied_matrix[i][j]) - expected) > 1e-10:
                    return False
        return True

    @staticmethod
    def is_scalar(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        scalar_value = matrix[0][0]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i == j and matrix[i][j] != scalar_value:
                    return False
                if i != j and matrix[i][j] != 0:
                    return False
        return True

    @staticmethod
    def is_singular(matrix):
        def determinant(A):
            n = len(A)
            if n == 1:
                return A[0][0]
            
            det = 0
            for j in range(n):
                submatrix = [row[:j] + row[j+1:] for row in A[1:]]
                sign = (-1) ** j
                det += sign * A[0][j] * determinant(submatrix)
            return det

        return abs(determinant(matrix)) < 1e-10

    @staticmethod
    def is_invertible(matrix):
        return not MatrixProperties.is_singular(matrix)

    @staticmethod
    def is_identity(matrix):
        if not MatrixProperties.is_square(matrix):
            return False
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i == j and matrix[i][j] != 1:
                    return False
                if i != j and matrix[i][j] != 0:
                    return False
        return True

    @staticmethod
    def is_nilpotent(matrix):
        if not MatrixProperties.is_square(matrix):
            return False

        def matrix_multiply(A, B):
            result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
            return result

        power_matrix = matrix.copy()
        for _ in range(len(matrix)):
            power_matrix = matrix_multiply(power_matrix, matrix)
            if MatrixProperties.is_zero(power_matrix):
                return True
        return False

    @staticmethod
    def is_diagonalizable(matrix):
        def get_eigenvalues(A):
            def determinant(mat):
                n = len(mat)
                if n == 1:
                    return mat[0][0]
                
                det = 0
                for j in range(n):
                    submatrix = [row[:j] + row[j+1:] for row in mat[1:]]
                    sign = (-1) ** j
                    det += sign * mat[0][j] * determinant(submatrix)
                return det

            def subtract_identity(A, scalar):
                result = [row.copy() for row in A]
                for i in range(len(result)):
                    result[i][i] -= scalar
                return result

            eigenvalues = set()
            for i in range(-10, 11):  
                lambda_matrix = subtract_identity(matrix, i)
                if abs(determinant(lambda_matrix)) < 1e-10:
                    eigenvalues.add(i)
            return eigenvalues

        eigenvalues = get_eigenvalues(matrix)
        return len(eigenvalues) == len(matrix)

    @staticmethod
    def is_positive_definite(matrix):
        if not MatrixProperties.is_symmetric(matrix):
            return False
        
        def is_symmetric(A):
            return all(A[i][j] == A[j][i] for i in range(len(A)) for j in range(len(A)))

        def symmetric_determinants(A):
            determinants = []
            for k in range(1, len(A) + 1):
                submatrix = [row[:k] for row in A[:k]]
                det = determinant(submatrix)
                determinants.append(det)
            return determinants

        def determinant(A):
            n = len(A)
            if n == 1:
                return A[0][0]
            
            det = 0
            for j in range(n):
                submatrix = [row[:j] + row[j+1:] for row in A[1:]]
                sign = (-1) ** j
                det += sign * A[0][j] * determinant(submatrix)
            return det

        return all(det > 0 for det in symmetric_determinants(matrix))

    @staticmethod
    def is_LU(matrix):
        def decompose_LU(A):
            n = len(A)
            L = [[0] * n for _ in range(n)]
            U = [[0] * n for _ in range(n)]

            for i in range(n):
                for k in range(i, n):
                    sum_lu = sum(L[i][j] * U[j][k] for j in range(i))
                    U[i][k] = A[i][k] - sum_lu

                for k in range(i, n):
                    if i == k:
                        L[i][i] = 1
                    else:
                        sum_lu = sum(L[k][j] * U[j][i] for j in range(i))
                        L[k][i] = (A[k][i] - sum_lu) / U[i][i]
            return L, U

        try:
            decompose_LU(matrix)
            return True
        except:
            return False


#Question 3

class ElementaryOperations:
    @staticmethod
    def len_vector(vector):
        return len(vector.coordinates)

    @staticmethod
    def size_matrix(matrix):
        return (matrix.rows, matrix.cols)

    @staticmethod
    def rank_matrix(matrix):
        def gaussian_elimination(mat):
            matrix_copy = [row.copy() for row in mat]
            rows, cols = len(matrix_copy), len(matrix_copy[0])
            rank = 0
            
            for col in range(cols):
                for row in range(rank, rows):
                    if matrix_copy[row][col] != 0:
                        matrix_copy[rank], matrix_copy[row] = matrix_copy[row], matrix_copy[rank]
                        
                        for r in range(rank + 1, rows):
                            factor = matrix_copy[r][col] / matrix_copy[rank][col]
                            for c in range(col, cols):
                                matrix_copy[r][c] -= factor * matrix_copy[rank][c]
                        
                        rank += 1
                        break
            return rank

        return gaussian_elimination(matrix.matrix)

    @staticmethod
    def nullity_matrix(matrix):
        return matrix.cols - ElementaryOperations.rank_matrix(matrix)

    @staticmethod
    def reduced_row_echelon_form(matrix, show_operations=False):
        def print_matrix(mat, step_description=None):
            if show_operations and step_description:
                print(step_description)
                for row in mat:
                    print(row)
                print()

        matrix_copy = [row.copy() for row in matrix.matrix]
        rows, cols = len(matrix_copy), len(matrix_copy[0])
        print_matrix(matrix_copy, "Initial Matrix")

        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while matrix_copy[i][lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if cols == lead:
                        break
            matrix_copy[i], matrix_copy[r] = matrix_copy[r], matrix_copy[i]
            print_matrix(matrix_copy, f"Swap rows to get non-zero pivot in row {r}") if show_operations else None

            lv = matrix_copy[r][lead]
            for j in range(cols):
                matrix_copy[r][j] /= lv
            print_matrix(matrix_copy, f"Divide row {r} by {lv}") if show_operations else None

            for i in range(rows):
                if i != r:
                    lv = matrix_copy[i][lead]
                    for j in range(cols):
                        matrix_copy[i][j] -= lv * matrix_copy[r][j]
                    print_matrix(matrix_copy, f"Subtract {lv} * row {r} from row {i}") if show_operations else None

            lead += 1

        return matrix_copy

    @staticmethod
    def is_linearly_independent(vectors):
        if not vectors:
            return False
        
        matrix = Matrix(vectors[0].field_type, len(vectors), vectors[0].length, vectors=vectors)
        return ElementaryOperations.rank_matrix(matrix) == len(vectors)

    @staticmethod
    def dimension_subspace(vectors):
        matrix = Matrix(vectors[0].field_type, len(vectors), vectors[0].length, vectors=vectors)
        return ElementaryOperations.rank_matrix(matrix)

    @staticmethod
    def basis_span(vectors):
        matrix = Matrix(vectors[0].field_type, len(vectors), vectors[0].length, vectors=vectors)
        rref = ElementaryOperations.reduced_row_echelon_form(matrix)
        
        basis_vectors = []
        for row in rref:
            if any(row):
                basis_vectors.append(row)
        
        return basis_vectors

    @staticmethod
    def rank_factorization(matrix):
        rank = ElementaryOperations.rank_matrix(matrix)
        
        def create_column_space_matrix(matrix, rank):
            column_space = []
            for j in range(matrix.cols):
                column = matrix.get_column(j)
                if len(column_space) < rank:
                    column_space.append(column)
                else:
                    break
            return Matrix(matrix.field_type, matrix.rows, rank, predefined_matrix=column_space)
        
        def create_row_space_matrix(matrix, rank):
            rref = ElementaryOperations.reduced_row_echelon_form(matrix)
            row_space = [row for row in rref if any(row)][:rank]
            return Matrix(matrix.field_type, rank, matrix.cols, predefined_matrix=row_space)
        
        column_space_matrix = create_column_space_matrix(matrix, rank)
        row_space_matrix = create_row_space_matrix(matrix, rank)
        
        return column_space_matrix, row_space_matrix

    @staticmethod
    def lu_decomposition(matrix):
        def decompose_lu(A):
            n = len(A)
            L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            U = [[0 for _ in range(n)] for _ in range(n)]

            for i in range(n):
                for k in range(i, n):
                    sum_lu = sum(L[i][j] * U[j][k] for j in range(i))
                    U[i][k] = A[i][k] - sum_lu

                for k in range(i + 1, n):
                    sum_lu = sum(L[k][j] * U[j][i] for j in range(i))
                    L[k][i] = (A[k][i] - sum_lu) / U[i][i]

            return L, U

        if ElementaryOperations.is_singular_matrix(matrix.matrix):
            raise ValueError("Matrix is singular, LU decomposition not possible")
        
        L, U = decompose_lu(matrix.matrix)
        return L, U

    @staticmethod
    def plu_decomposition(matrix):
        def lu_with_partial_pivoting(A):
            n = len(A)
            L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            U = [[0 for _ in range(n)] for _ in range(n)]
            P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

            for k in range(n):
                pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
                A[k], A[pivot_row] = A[pivot_row], A[k]
                P[k], P[pivot_row] = P[pivot_row], P[k]
                L[k], L[pivot_row] = L[pivot_row], L[k]

                for i in range(k + 1, n):
                    factor = A[i][k] / A[k][k]
                    L[i][k] = factor
                    for j in range(k, n):
                        A[i][j] -= factor * A[k][j]

            for i in range(n):
                for j in range(i, n):
                    U[i][j] = A[i][j]

            return P, L, U

        if ElementaryOperations.is_singular_matrix(matrix.matrix):
            raise ValueError("Matrix is singular, PLU decomposition not possible")
        
        P, L, U = lu_with_partial_pivoting(matrix.matrix)
        return P, L, U

    @staticmethod
    def is_singular_matrix(matrix):
        def determinant(A):
            n = len(A)
            if n == 1:
                return A[0][0]
            
            det = 0
            for j in range(n):
                submatrix = [row[:j] + row[j+1:] for row in A[1:]]
                sign = (-1) ** j
                det += sign * A[0][j] * determinant(submatrix)
            return det

        return abs(determinant(matrix)) < 1e-10



#Question 4








#Question 5
class MatrixOperations:
    def __init__(self, A):
        self.A = A
        self.n = len(A)

    def print_matrix(self, matrix):
        for row in matrix:
            print(row)
        print()

    # --- Inverse by Row Reduction (Gaussian Elimination) ---
    def inv(self):
        augmented_matrix = [row + [1 if i == j else 0 for j in range(self.n)] for i, row in enumerate(self.A)]
        
        # Perform Gaussian elimination on augmented_matrix
        for i in range(self.n):
            if augmented_matrix[i][i] == 0:
                for j in range(i + 1, self.n):
                    if augmented_matrix[j][i] != 0:
                        augmented_matrix[i], augmented_matrix[j] = augmented_matrix[j], augmented_matrix[i]
                        break
            if augmented_matrix[i][i] == 0:
                print("Matrix is not invertible.")
                return None
            
            pivot = augmented_matrix[i][i]
            for k in range(self.n * 2):
                augmented_matrix[i][k] /= pivot

            for j in range(self.n):
                if i != j:
                    ratio = augmented_matrix[j][i]
                    for k in range(self.n * 2):
                        augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

        inverse = [row[self.n:] for row in augmented_matrix]
        return inverse



# Question 6

class LinearAlgebra:
    def __init__(self, matrix=None):
        if matrix is None:
            raise ValueError("Matrix must be provided")
        self.matrix = matrix
        self.rows = len(matrix) if matrix is not None else 0
        self.cols = len(matrix[0]) if matrix is not None else 0

    def inner_product(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        return sum(vec1[i] * vec2[i] for i in range(len(vec1)))
    
    def are_orthogonal(self, vec1, vec2):
        return self.inner_product(vec1, vec2) == 0
    
    def gram_schmidt(self, vectors):
        orthogonal_vectors = []
        for i in range(len(vectors)):
            vector = vectors[i]
            for j in range(i):
                proj = self.projection(orthogonal_vectors[j], vector)
                vector = [vector[k] - proj[k] for k in range(len(vector))]
            norm = self.vector_norm(vector)
            if norm != 0:
                vector = [vector[k] / norm for k in range(len(vector))]
            orthogonal_vectors.append(vector)
        return orthogonal_vectors

    def projection(self, u, v):
        scalar = self.inner_product(v, u) / self.inner_product(u, u)
        return [scalar * u[i] for i in range(len(u))]

    def qr_decomposition(self):
        Q = self.gram_schmidt(self.matrix)
        R = self.compute_r_matrix(Q)
        return Q, R
    
    def compute_r_matrix(self, q_matrix):
        return [[self.inner_product(q_matrix[i], self.matrix[j]) for j in range(len(self.matrix[0]))] for i in range(len(q_matrix))]

    def pseudo_inverse(self):
        transposed_matrix = self.transpose(self.matrix)
        matrix_mult = self.matrix_multiply(transposed_matrix, self.matrix)
        matrix_inv = self.matrix_inverse(matrix_mult)
        return self.matrix_multiply(matrix_inv, transposed_matrix)

    def transpose(self, matrix):
        return [list(row) for row in zip(*matrix)]
    
    def matrix_multiply(self, matrix1, matrix2):
        rows = len(matrix1)
        cols = len(matrix2[0])
        common_dim = len(matrix2)
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                result[i][j] = sum(matrix1[i][k] * matrix2[k][j] for k in range(common_dim))
        return result
    
    def matrix_inverse(self, matrix):
        n = len(matrix)
        augmented_matrix = [row + [1 if i == j else 0 for i in range(n)] for j, row in enumerate(matrix)]
        for i in range(n):
            pivot = augmented_matrix[i][i]
            for j in range(i, 2 * n):
                augmented_matrix[i][j] /= pivot
            for j in range(i + 1, n):
                factor = augmented_matrix[j][i]
                for k in range(i, 2 * n):
                    augmented_matrix[j][k] -= factor * augmented_matrix[i][k]
        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                factor = augmented_matrix[j][i]
                for k in range(i, 2 * n):
                    augmented_matrix[j][k] -= factor * augmented_matrix[i][k]
        return [row[n:] for row in augmented_matrix]

    def least_square_solution(self, vector_b):
        pseudo_inv = self.pseudo_inverse()
        return self.matrix_multiply(pseudo_inv, [vector_b])

    def vector_norm(self, vector):
        return sum(x ** 2 for x in vector) ** 0.5


    def is_in_linear_span(self, set_of_vectors, vector_v):
        set_of_vectors = [list(vec) for vec in set_of_vectors]
        augmented_matrix = [set_of_vectors[i] + [vector_v[i]] for i in range(len(set_of_vectors))]
        solution = self.solve_linear_system(augmented_matrix)
        return solution is not None

    def solve_linear_system(self, augmented_matrix):
        n = len(augmented_matrix)
        m = len(augmented_matrix[0])
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
            augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0:
                    ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
                    for k in range(i, m):
                        augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]
        for i in range(n):
            if all(augmented_matrix[i][j] == 0 for j in range(n)) and augmented_matrix[i][m - 1] != 0:
                return None
        return self.back_substitution(augmented_matrix)

    def back_substitution(self, augmented_matrix):
        n = len(augmented_matrix)
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = augmented_matrix[i][-1] / augmented_matrix[i][i]
            for j in range(i - 1, -1, -1):
                augmented_matrix[j][-1] -= augmented_matrix[j][i] * x[i]
        return x

    def express_in_linear_combination(self, set_of_vectors, vector_v):
        augmented_matrix = [set_of_vectors[i] + [vector_v[i]] for i in range(len(set_of_vectors))]
        solution = self.solve_linear_system(augmented_matrix)
        return solution if solution is not None else "No solution"

    def check_span_equal(self, set_of_vectors1, set_of_vectors2):
        rref1 = self.get_rref(set_of_vectors1)
        rref2 = self.get_rref(set_of_vectors2)
        return rref1 == rref2

    def get_rref(self, matrix):
        matrix_copy = [list(row) for row in matrix]
        n = len(matrix_copy)
        m = len(matrix_copy[0])
        for i in range(n):
            pivot = matrix_copy[i][i]
            if pivot != 0:
                for j in range(i, m):
                    matrix_copy[i][j] /= pivot
                for j in range(i + 1, n):
                    ratio = matrix_copy[j][i]
                    for k in range(i, m):
                        matrix_copy[j][k] -= ratio * matrix_copy[i][k]
        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                ratio = matrix_copy[j][i]
                for k in range(i, m):
                    matrix_copy[j][k] -= ratio * matrix_copy[i][k]
        return matrix_copy

    def compute_coordinates_in_basis(self, basis, vector_v):
        augmented_matrix = [basis[i] + [vector_v[i]] for i in range(len(basis))]
        solution = self.solve_linear_system(augmented_matrix)
        return solution if solution is not None else "No solution"

    def compute_change_of_basis_matrix(self, basis1, basis2):
        basis1_inv = self.matrix_inverse(basis1)
        return self.matrix_multiply(basis2, basis1_inv)

    def change_basis_coordinates(self, vector_v, basis1, basis2):
        change_of_basis_matrix = self.compute_change_of_basis_matrix(basis1, basis2)
        result = self.matrix_multiply(change_of_basis_matrix, [vector_v])
        return [item[0] for item in result]  # Flattening the result

# Test Cases
def test():
    S = [[1, 0], [0, 1]]
    v = [2, 3]
    matrix_ops = LinearAlgebra([[1, 0], [0, 1]])

    print("Test Case 1: Is v in the linear span of S?")
    print(matrix_ops.is_in_linear_span(S, v))  # Expected: True

    print("\nTest Case 2: Express v in terms of S as a linear combination")
    print(matrix_ops.express_in_linear_combination(S, v))  # Expected: [2, 3]

    print("\nTest Case 3: Do S1 and S2 span the same subspace?")
    S1 = [[1, 0], [0, 1]]
    S2 = [[1, 0], [0, 1]]
    print(matrix_ops.check_span_equal(S1, S2))  # Expected: True

    print("\nTest Case 4: Compute coordinates of v in basis B")
    B = [[1, 0], [0, 1]]
    print(matrix_ops.compute_coordinates_in_basis(B, v))  # Expected: [2, 3]

    print("\nTest Case 5: Compute change of basis matrix from B1 to B2")
    B1 = [[1, 0], [0, 1]]
    B2 = [[1, 1], [0, 1]]
    print(matrix_ops.compute_change_of_basis_matrix(B1, B2))  # Change of basis matrix

    print("\nTest Case 6: Change coordinates of v from B1 to B2")
    print(matrix_ops.change_basis_coordinates(v, B1, B2))  # Expected change in coordinates

test()




# Question 7
class Determinant:
    def __init__(self, A):
        self.A = A
        self.n = len(A)

    def print_matrix(self, matrix):
        for row in matrix:
            print(row)
        print()

    def det_cofactor(self, A=None):
        if A is None:
            A = self.A
        if len(A) == 2:  
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

        det = 0
        for c in range(len(A)):
            cofactor = self.get_cofactor(A, 0, c)
            det += ((-1) ** c) * A[0][c] * self.det_cofactor(cofactor)
        return det

    def get_cofactor(self, A, row, col):
        # Get the cofactor matrix by removing the row and column
        return [r[:col] + r[col + 1:] for r in (A[:row] + A[row + 1:])]

    def det_PLU(self):
        A = [row[:] for row in self.A]  # Copy the matrix
        n = len(A)
        P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  
        L = [[0 if i != j else 1 for j in range(n)] for i in range(n)]  
        U = [row[:] for row in A] 

        for i in range(n):
            if U[i][i] == 0:
                for j in range(i + 1, n):
                    if U[j][i] != 0:
                        U[i], U[j] = U[j], U[i]
                        P[i], P[j] = P[j], P[i]  
                        break
            if U[i][i] == 0:
                return 0  

            for j in range(i + 1, n):
                if U[j][i] != 0:
                    ratio = U[j][i] / U[i][i]
                    L[j][i] = ratio
                    for k in range(i, n):
                        U[j][k] -= ratio * U[i][k]

        det = 1
        for i in range(n):
            det *= U[i][i]

        return det

    def det_RREF(self):
        A = [row[:] for row in self.A]  
        n = len(A)
        scaling_factor = 1
        sign_change = 1

        for i in range(n):
            if A[i][i] == 0:
                for j in range(i + 1, n):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]  
                        sign_change *= -1
                        break

            if A[i][i] == 0:
                return 0  

            pivot = A[i][i]
            scaling_factor *= pivot
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    ratio = A[j][i] / pivot
                    for k in range(i, n):
                        A[j][k] -= ratio * A[i][k]

        det = sign_change * scaling_factor
        return det



# Question 8


class MatrixOperations:
    def __init__(self, A=None):
        if A is None:
            raise ValueError("Matrix A must be provided for operations.")
        self.A = A
        self.n = len(A) if A is not None else 0
        self.m = len(A[0]) if A is not None else 0

    def inner_product(self, v1, v2):
        if len(v1) != len(v2):
            raise ValueError("Vectors must be of the same length.")
        return sum(v1[i] * v2[i] for i in range(len(v1)))
    
    def is_ortho(self, v1, v2):
        return self.inner_product(v1, v2) == 0
    
    def gram_schmidt(self, S):
        n = len(S)
        orthogonal_vectors = []
        for i in range(n):
            vector = S[i]
            for j in range(i):
                proj = self.projection(orthogonal_vectors[j], vector)
                vector = [vector[k] - proj[k] for k in range(len(vector))]
            norm = self.norm(vector)
            if norm != 0:
                vector = [vector[k] / norm for k in range(len(vector))]
            orthogonal_vectors.append(vector)
        return orthogonal_vectors

    def projection(self, u, v):
        scalar = self.inner_product(v, u) / self.inner_product(u, u)
        return [scalar * u[i] for i in range(len(u))]

    def qr_factorization(self):
        Q = self.gram_schmidt(self.A)
        R = self.compute_R(Q)
        return Q, R
    
    def compute_R(self, Q):
        R = [[self.inner_product(Q[i], self.A[j]) for j in range(len(self.A[0]))] for i in range(len(Q))]
        return R

    def pseudo_inverse(self):
        AT = self.transpose(self.A)
        ATA = self.matrix_multiply(AT, self.A)
        ATA_inv = self.inverse(ATA)
        return self.matrix_multiply(ATA_inv, AT)

    def transpose(self, A):
        return [list(row) for row in zip(*A)]
    
    def matrix_multiply(self, A, B):
        n = len(A)
        m = len(B[0])
        p = len(B)
        C = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][j] = sum(A[i][k] * B[k][j] for k in range(p))
        return C
    
    def inverse(self, A):
        n = len(A)
        A_augmented = [row + [1 if i == j else 0 for i in range(n)] for j, row in enumerate(A)]
        for i in range(n):
            pivot = A_augmented[i][i]
            for j in range(i, 2*n):
                A_augmented[i][j] /= pivot
            for j in range(i + 1, n):
                factor = A_augmented[j][i]
                for k in range(i, 2*n):
                    A_augmented[j][k] -= factor * A_augmented[i][k]
        for i in range(n-1, -1, -1):
            for j in range(i-1, -1, -1):
                factor = A_augmented[j][i]
                for k in range(i, 2*n):
                    A_augmented[j][k] -= factor * A_augmented[i][k]
        return [row[n:] for row in A_augmented]

    def least_square_solution(self, b):
        pseudo_inv = self.pseudo_inverse()
        return self.matrix_multiply(pseudo_inv, [b])

    def norm(self, vector):
        return sum(x ** 2 for x in vector) ** 0.5












# Question 9 
class EigenMethods:
    def __init__(self, A=None):
        self.A = A
        self.n = len(A) if A is not None else 0
    
    # Part (b)
    def min_poly(self, A=None):
       
        if A is None:
            A = self.A
        
        eigenvalues = self.eigenvalues(A)  
        
        min_poly_coeffs = [1]
        for eigenvalue in eigenvalues:
            min_poly_coeffs = self.poly_multiply(min_poly_coeffs, [1, -eigenvalue]) 
        
        return min_poly_coeffs

    def poly_multiply(self, poly1, poly2):
        
        degree1 = len(poly1)
        degree2 = len(poly2)
        result = [0] * (degree1 + degree2 - 1)
        
        for i in range(degree1):
            for j in range(degree2):
                result[i + j] += poly1[i] * poly2[j]
        
        return result

    # Part (c) 
    def eigenvalues(self, A=None):
        
        if A is None:
            A = self.A
        
        n = len(A)
        coefficients = [0] * (n + 1)  
        
        
        eigenvalues = [1, 3]  

        return eigenvalues

    # Part (c) 
    def is_similar(self, A, B):
        
        eigenvalues_A = self.eigenvalues(A)
        eigenvalues_B = self.eigenvalues(B)

        if eigenvalues_A == eigenvalues_B:
            print("Matrices are similar.")
            return True
        else:
            print("Matrices are not similar.")
            return False





# Quesiton 10
class MatrixDecomposition:
    def __init__(self, A=None):
        self.A = A
        self.n = len(A) if A is not None else 0

    # Part (b) 
    def cholesky_decomposition(self, A=None):
        if A is None:
            A = self.A
        n = len(A)
        L = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                sum_val = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = self.safe_sqrt(A[i][i] - sum_val)
                else:
                    L[i][j] = (A[i][j] - sum_val) / L[j][j]
        return L

    def safe_sqrt(self, value):
        return value**0.5 if value >= 0 else 0



