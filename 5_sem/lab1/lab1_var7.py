# variant7.py
"""
Вариант 7:
1) Метод квадратного корня (Cholesky) — собственная реализация;
2) Метод отражений (Householder) — вручную формируем отражения, получаем R, решаем;
3) Метод релаксации (SOR) — собственная реализация;
4) Метод минимальных невязок (алгоритм минимизации невязки по направлению r).

Вход: n с клавиатуры.
Генерация: симметричная положительно-определённая матрица A через M*M^T + n*I,
элементы с 2 знаками.
Все операции вручную (без NumPy).
Аккуратная табличная печать.
"""

import random
import math
import copy

# ---------- Вспомогательные функции (аналогично variant4) ----------

def fmt(x):
    return f"{x:12.4f}"

def print_matrix(A, name="A"):
    print(f"\n{name}:")
    for row in A:
        print(" ".join(fmt(v) for v in row))
    print()

def print_vector(v, name="v"):
    print(f"\n{name}:")
    for val in v:
        print(fmt(val))
    print()

def zeros(n, m=None):
    if m is None:
        return [0.0]*n
    return [[0.0]*m for _ in range(n)]

def copy_mat(A):
    return [row[:] for row in A]

def mat_vec_mul(A, x):
    n = len(A)
    m = len(A[0])
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += A[i][j] * x[j]
        y[i] = s
    return y

def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i]*b[i]
    return s

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def scalar_vec_mul(alpha, v):
    return [alpha * vi for vi in v]

def norm2(v):
    return math.sqrt(dot(v, v))

def transpose(A):
    n = len(A); m = len(A[0])
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T

def inf_norm_mat(A):
    best = 0.0
    for row in A:
        s = sum(abs(x) for x in row)
        if s > best:
            best = s
    return best

# ---------- Генерация СПД матрицы и вектора ----------

def gen_spd_matrix(n):
    # создаём случайную матрицу M, затем A = M*M^T + n*I (гарантированно SPD)
    M = zeros(n, n)
    for i in range(n):
        for j in range(n):
            M[i][j] = round(random.uniform(-5, 5), 2)
    # A = M * M^T
    A = zeros(n, n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += M[i][k]*M[j][k]
            A[i][j] = s
    # добавим n на диагональ для улучшения обусловленности и положительной определённости
    for i in range(n):
        A[i][i] = round(A[i][i] + n, 2)
    # округлим остальные элементы до 2 знаков
    for i in range(n):
        for j in range(n):
            A[i][j] = round(A[i][j], 2)
    return A

def gen_vector(n):
    return [round(random.uniform(-5, 5), 2) for _ in range(n)]

# ---------- 1) Метод квадратного корня (Cholesky) ----------

def cholesky_decompose(A):
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s += L[i][k]*L[j][k]
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise Exception("Матрица не положительно определена (в процессе Cholesky).")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_cholesky(A, b):
    L = cholesky_decompose(A)
    n = len(A)
    # прямой ход: L y = b
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i][j]*y[j]
        y[i] = (b[i] - s)/L[i][i]
    # обратный ход: L^T x = y
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s += L[j][i]*x[j]  # L^T[i][j] = L[j][i]
        x[i] = (y[i] - s)/L[i][i]
    return x, L

# ---------- 2) Метод отражений (Householder QR) ----------

def householder_qr_solve(A_init, b_init):
    A = copy_mat(A_init)
    b = b_init[:]
    n = len(A)
    # Для квадратной A применим n-1 отражение
    for k in range(n-1):
        # формируем вектор x = A[k:, k]
        x = [A[i][k] for i in range(k, n)]
        # нормируем и создаём вектор v
        normx = math.sqrt(sum(xi*xi for xi in x))
        if normx < 1e-14:
            continue
        # e1
        e1 = [0.0]*len(x)
        e1[0] = 1.0
        sign = 1.0 if x[0] >= 0 else -1.0
        v = [x[i] + sign*normx*e1[i] for i in range(len(x))]
        v_norm = math.sqrt(sum(vi*vi for vi in v))
        if v_norm < 1e-14:
            continue
        v = [vi / v_norm for vi in v]
        # применяем отражение к подматрице A[k:, k:]
        for i in range(k, n):
            for j in range(k, n):
                # A[i][j] -= 2 * v[i-k] * (v dot A[k:, j])
                dot_v_col = 0.0
                for t in range(len(v)):
                    dot_v_col += v[t] * A[k + t][j]
                A[i][j] = A[i][j] - 2.0 * v[i-k] * dot_v_col
        # применяем к вектору b
        for i in range(k, n):
            dot_v_b = 0.0
            for t in range(len(v)):
                dot_v_b += v[t] * b[k + t]
            b[i] = b[i] - 2.0 * v[i-k] * dot_v_b
    # теперь A верхняя треугольная (R)
    x = [0.0]*n
    # обратная подстановка R x = b'
    for i in range(n-1, -1, -1):
        if abs(A[i][i]) < 1e-14:
            x[i] = 0.0
            continue
        s = 0.0
        for j in range(i+1, n):
            s += A[i][j]*x[j]
        x[i] = (b[i] - s)/A[i][i]
    return x, A  # возвращаем также R для вывода

# ---------- 3) Метод релаксации (SOR) ----------

def sor(A, b, w=1.1, tol=1e-8, max_iter=10000):
    n = len(A)
    x = [0.0]*n
    for it in range(max_iter):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j]*x_new[j] for j in range(i))
            s2 = sum(A[i][j]*x[j] for j in range(i+1, n))
            x_new[i] = (1 - w)*x[i] + (w / A[i][i])*(b[i] - s1 - s2)
        if norm2(vec_sub(x_new, x)) < tol:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ---------- 4) Метод минимальных невязок (Minimal Residuals) ----------

def minimal_residuals(A, b, tol=1e-8, max_iter=10000):
    n = len(A)
    x = [0.0]*n
    for it in range(max_iter):
        r = vec_sub(b, mat_vec_mul(A, x))
        Ar = mat_vec_mul(A, r)
        denom = dot(Ar, Ar)
        if abs(denom) < 1e-20:
            break
        alpha = dot(r, Ar) / denom
        x_new = vec_add(x, scalar_vec_mul(alpha, r))
        if norm2(vec_sub(x_new, x)) < tol:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ---------- Обратная матрица через Gauss-Jordan (для числа обусловленности) ----------

def inverse_gauss_jordan(A_init):
    A = copy_mat(A_init)
    n = len(A)
    aug = [row[:] + [1.0 if i==j else 0.0 for j in range(n)] for i,row in enumerate(A)]
    for i in range(n):
        max_row = i
        max_val = abs(aug[i][i])
        for r in range(i+1, n):
            if abs(aug[r][i]) > max_val:
                max_val = abs(aug[r][i])
                max_row = r
        if max_val < 1e-14:
            raise Exception("Матрица вырождена.")
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]
        pivot = aug[i][i]
        # нормируем
        for col in range(2*n):
            aug[i][col] /= pivot
        # зануляем остальные
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor == 0:
                continue
            for col in range(2*n):
                aug[r][col] -= factor * aug[i][col]
    inv = [row[n:] for row in aug]
    return inv

# ---------- Основная программа ----------

def main():
    random.seed()
    n = int(input("Введите размерность системы n: "))

    A = gen_spd_matrix(n)
    x_true = gen_vector(n)
    b = mat_vec_mul(A, x_true)

    print_matrix(A, "Матрица A (симметричная, положительно определённая)")
    print_vector(x_true, "Истинный вектор x_true")
    print_vector(b, "Вектор b = A * x_true")

    # 1) Cholesky
    print("\n=== 1) Метод квадратного корня (Cholesky) ===")
    try:
        x_chol, L = solve_cholesky(A, b)
        print_matrix(L, "Матрица L (нижняя треугольная) из Cholesky")
        print_vector(x_chol, "Решение x (Cholesky)")
        res_chol = vec_sub(b, mat_vec_mul(A, x_chol))
        print_vector(res_chol, "Невязка r (Cholesky)")
        print(f"Норма невязки ||r|| = {norm2(res_chol):.6e}")
    except Exception as e:
        print("Cholesky не выполнился:", e)

    # 2) Householder
    print("\n=== 2) Метод отражений (Householder QR) ===")
    x_house, R = householder_qr_solve(A, b)
    print_matrix(R, "Матрица R (верхняя треугольная) из Householder")
    print_vector(x_house, "Решение x (Householder)")
    res_house = vec_sub(b, mat_vec_mul(A, x_house))
    print_vector(res_house, "Невязка r (Householder)")
    print(f"Норма невязки ||r|| = {norm2(res_house):.6e}")

    # 3) SOR
    print("\n=== 3) Метод релаксации (SOR) ===")
    w = 1.1
    x_sor, its_sor = sor(A, b, w=w)
    print_vector(x_sor, f"Решение x (SOR), w={w}, итераций={its_sor}")
    res_sor = vec_sub(b, mat_vec_mul(A, x_sor))
    print_vector(res_sor, "Невязка r (SOR)")
    print(f"Норма невязки ||r|| = {norm2(res_sor):.6e}")

    # 4) Минимальные невязки
    print("\n=== 4) Метод минимальных невязок ===")
    x_minres, its_minres = minimal_residuals(A, b)
    print_vector(x_minres, f"Решение x (MinRes), итераций={its_minres}")
    res_minres = vec_sub(b, mat_vec_mul(A, x_minres))
    print_vector(res_minres, "Невязка r (MinRes)")
    print(f"Норма невязки ||r|| = {norm2(res_minres):.6e}")

    # Обратная матрица и число обусловленности (infty-норма)
    try:
        A_inv = inverse_gauss_jordan(A)
        print_matrix(A_inv, "Обратная матрица A^{-1} (Gauss-Jordan)")
        cond = inf_norm_mat(A) * inf_norm_mat(A_inv)
        print(f"Число обусловленности (infty-норма) cond = {cond:.6e}")
    except Exception as e:
        print("Не удалось найти обратную матрицу:", e)

if __name__ == "__main__":
    main()
