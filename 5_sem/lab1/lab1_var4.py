# variant4.py
"""
Вариант 4:
1) Метод Гаусса с выбором главного элемента по столбцу (реализация вручную).
2) Метод вращений (используем Givens-вращения для зануления поддиагональных элементов,
   получаем верхнюю треугольную матрицу, затем обратный ход).
3) Метод Зейделя (итерационный).
4) Метод градиентного спуска (степеньший спуск для квадратичной формы).

Вход: n с клавиатуры.
Генерация: случайная диагонально-доминирующая матрица A (числа с 2 знаками после запятой),
случайный истинный вектор x_true (2 знака).
Вычисляем b = A * x_true.
Все операции реализованы вручную (списки Python).
Красивая табличная печать результатов.
"""

import random
import math
import copy

# ---------- Вспомогательные функции ----------

def fmt(x):
    """Формат одного числа для вывода (ширина 12, 4 знака после запятой)."""
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
        return [0.0] * n
    return [[0.0]*m for _ in range(n)]

def copy_mat(A):
    return [row[:] for row in A]

def transpose(A):
    n = len(A)
    m = len(A[0])
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T

def mat_vec_mul(A, x):
    n = len(A)
    m = len(A[0])
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += A[i][j]*x[j]
        y[i] = s
    return y

def mat_mul(A, B):
    n = len(A)
    p = len(B[0])
    m = len(B)
    C = zeros(n, p)
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def scalar_vec_mul(alpha, v):
    return [alpha * vi for vi in v]

def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i]*b[i]
    return s

def norm2(v):
    return math.sqrt(dot(v, v))

def inf_norm_mat(A):
    # infinity norm (max row sum)
    n = len(A)
    best = 0.0
    for i in range(n):
        s = sum(abs(x) for x in A[i])
        if s > best:
            best = s
    return best

# ---------- Генерация данных ----------

def gen_diag_dominant(n):
    A = zeros(n, n)
    for i in range(n):
        for j in range(n):
            A[i][j] = round(random.uniform(-5, 5), 2)
    # сделаем диагональное доминирование по строке
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = round(row_sum + random.uniform(1.0, 5.0), 2)
    return A

def gen_vector(n):
    return [round(random.uniform(-5, 5), 2) for _ in range(n)]

# ---------- 1) Гаусс с выбором главного по столбцу ----------

def gaussian_column_pivot(A_init, b_init):
    A = copy_mat(A_init)
    b = b_init[:]  # copy
    n = len(A)
    det_sign = 1
    det = 1.0
    # прямой ход
    for i in range(n):
        # найти строку с макс модулем в столбце i на местах i..n-1
        max_row = i
        max_val = abs(A[i][i])
        for r in range(i+1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                max_row = r
        if max_val < 1e-14:
            # нулевой столбец -> детерминант ноль
            det = 0.0
            continue
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
            det_sign *= -1
        pivot = A[i][i]
        det *= pivot
        # исключаем
        for j in range(i+1, n):
            if A[j][i] == 0:
                continue
            factor = A[j][i] / pivot
            # A[j][k] = A[j][k] - factor * A[i][k] для k=i..n-1
            for k in range(i, n):
                A[j][k] = A[j][k] - factor * A[i][k]
            b[j] = b[j] - factor * b[i]
    # обратный ход (обратная подстановка)
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(A[i][i]) < 1e-14:
            x[i] = 0.0
            continue
        s = 0.0
        for j in range(i+1, n):
            s += A[i][j]*x[j]
        x[i] = (b[i] - s)/A[i][i]
    det *= det_sign
    return x, det, A, b

# ---------- Обратная матрица через Gauss-Jordan (напрямую) ----------
def inverse_gauss_jordan(A_init):
    A = copy_mat(A_init)
    n = len(A)
    # расширяем A | I
    aug = [row[:] + [1.0 if i==j else 0.0 for j in range(n)] for i,row in enumerate(A)]
    # приводим до I | A^{-1}
    for i in range(n):
        # выбор главного по столбцу (включая i..n-1)
        max_row = i
        max_val = abs(aug[i][i])
        for r in range(i+1, n):
            if abs(aug[r][i]) > max_val:
                max_val = abs(aug[r][i])
                max_row = r
        if max_val < 1e-14:
            raise Exception("Матрица вырождена, обратной нет.")
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]
        # нормируем строку i
        pivot = aug[i][i]
        for col in range(2*n):
            aug[i][col] /= pivot
        # зануляем все остальные строки по столбцу i
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            if factor == 0:
                continue
            for col in range(2*n):
                aug[r][col] -= factor * aug[i][col]
    # извлекаем правую часть
    inv = [row[n:] for row in aug]
    return inv

# ---------- 2) Метод вращений (Givens) - триангуляция A, меняем и b ----------

def givens_triangularize(A_init, b_init):
    A = copy_mat(A_init)
    b = b_init[:]
    n = len(A)
    # Перебираем столбцы, для каждого столбца зануляем элементы ниже диагонали
    for j in range(n):
        for i in range(j+1, n):
            a = A[j][j]
            b_elem = A[i][j]
            if abs(b_elem) < 1e-14:
                continue
            r = math.hypot(a, b_elem)  # sqrt(a^2 + b^2) реализовано через hypot
            c = a / r
            s = -b_elem / r
            # Применяем G к строкам j и i: [row_j; row_i] = G * [row_j; row_i]
            # где G = [[c, -s],[s,c]] действует на первые элементы от j до n-1
            for col in range(j, n):
                temp_j = c*A[j][col] - s*A[i][col]
                temp_i = s*A[j][col] + c*A[i][col]
                A[j][col] = temp_j
                A[i][col] = temp_i
            # и к вектору b
            temp_bj = c*b[j] - s*b[i]
            temp_bi = s*b[j] + c*b[i]
            b[j] = temp_bj
            b[i] = temp_bi
    # после всех G A верхняя треугольная
    return A, b

def back_substitution_upper(R, y):
    n = len(R)
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        if abs(R[i][i]) < 1e-14:
            x[i] = 0.0
            continue
        s = 0.0
        for j in range(i+1, n):
            s += R[i][j]*x[j]
        x[i] = (y[i] - s)/R[i][i]
    return x

# ---------- 3) Метод Зейделя (iterative) ----------

def seidel(A, b, tol=1e-8, max_iter=10000):
    n = len(A)
    x = [0.0]*n
    for it in range(max_iter):
        x_new = x[:]  # copy
        for i in range(n):
            s1 = sum(A[i][j]*x_new[j] for j in range(i))
            s2 = sum(A[i][j]*x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2)/A[i][i]
        # check
        diff = [x_new[i]-x[i] for i in range(n)]
        if norm2(diff) < tol:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ---------- 4) Метод градиентного спуска (Steepest Descent) ----------

def gradient_descent_method(A, b, tol=1e-8, max_iter=10000):
    n = len(A)
    x = [0.0]*n
    for it in range(max_iter):
        r = vec_sub(b, mat_vec_mul(A, x))    # r = b - A x
        Ar = mat_vec_mul(A, r)
        denom = dot(r, Ar)
        if abs(denom) < 1e-20:
            break
        alpha = dot(r, r) / denom
        x_new = vec_add(x, scalar_vec_mul(alpha, r))
        if norm2(vec_sub(x_new, x)) < tol:
            return x_new, it+1
        x = x_new
    return x, max_iter

# ---------- Основная программа ----------

def main():
    random.seed()
    n = int(input("Введите размерность системы n: "))

    A = gen_diag_dominant(n)
    x_true = gen_vector(n)
    b = mat_vec_mul(A, x_true)

    print_matrix(A, "Матрица A (диагонально-доминирующая)")
    print_vector(x_true, "Истинный вектор x_true")
    print_vector(b, "Вектор b = A * x_true")

    # Гаусс с выбором главного по столбцу
    print("\n=== 1) Метод Гаусса с выбором главного элемента по столбцу ===")
    x_gauss, det_gauss, A_after, b_after = gaussian_column_pivot(A, b)
    print_vector(x_gauss, "Решение x (Гаусс)")
    print(f"Детерминант (из процедур) = {det_gauss:.6e}")
    # вычислим невязку
    res_gauss = vec_sub(b, mat_vec_mul(A, x_gauss))
    print_vector(res_gauss, "Невязка r = b - A*x (Гаусс) (вектор)")
    print(f"Норма невязки ||r|| = {norm2(res_gauss):.6e}")

    # Обратная матрица (Gauss-Jordan) и число обусловленности (через infinity norm)
    try:
        A_inv = inverse_gauss_jordan(A)
        print_matrix(A_inv, "Обратная матрица A^{-1} (Gauss-Jordan)")
        cond = inf_norm_mat(A) * inf_norm_mat(A_inv)
        print(f"Число обусловленности (infty-норма) cond = {cond:.6e}")
    except Exception as e:
        print("Не удалось найти обратную матрицу:", e)

    # Метод вращений (Givens)
    print("\n=== 2) Метод вращений (Givens) для триангуляции и решения ===")
    R, b_rot = givens_triangularize(A, b)
    print_matrix(R, "Матрица R (верхняя треугольная) после Givens")
    print_vector(b_rot, "Вектор b' после применения Givens")
    x_givens = back_substitution_upper(R, b_rot)
    print_vector(x_givens, "Решение x (Givens)")
    res_givens = vec_sub(b, mat_vec_mul(A, x_givens))
    print_vector(res_givens, "Невязка r (Givens)")
    print(f"Норма невязки ||r|| = {norm2(res_givens):.6e}")

    # Метод Зейделя
    print("\n=== 3) Метод Зейделя (итерационный) ===")
    x_seidel, it_seidel = seidel(A, b)
    print_vector(x_seidel, f"Решение x (Seidel), итераций {it_seidel}")
    res_seidel = vec_sub(b, mat_vec_mul(A, x_seidel))
    print_vector(res_seidel, "Невязка r (Seidel)")
    print(f"Норма невязки ||r|| = {norm2(res_seidel):.6e}")

    # Метод градиентного спуска
    print("\n=== 4) Метод градиентного спуска (Steepest Descent) ===")
    x_gd, it_gd = gradient_descent_method(A, b)
    print_vector(x_gd, f"Решение x (GradDesc), итераций {it_gd}")
    res_gd = vec_sub(b, mat_vec_mul(A, x_gd))
    print_vector(res_gd, "Невязка r (GradDesc)")
    print(f"Норма невязки ||r|| = {norm2(res_gd):.6e}")

if __name__ == "__main__":
    main()
