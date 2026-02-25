import math


def general_cholesky_decomposition(A):
    n = len(A)

    # Проверка симметричности
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-10:
                raise ValueError(
                    f"Матрица не симметрична: A[{i}][{j}] = {A[i][j]}, A[{j}][{i}] = {A[j][i]}"
                )

    S = [[0.0] * n for _ in range(n)]
    D = [0] * n

    for i in range(n):
        sum_sq = 0.0
        for k in range(i):
            sum_sq += S[k][i] * S[k][i] * D[k]

        value = A[i][i] - sum_sq
        if abs(value) < 1e-10:
            raise ValueError("Матрица вырождена")

        D[i] = 1 if value >= 0 else -1
        S[i][i] = math.sqrt(abs(value))

        for j in range(i + 1, n):
            sum_prod = 0.0
            for k in range(i):
                sum_prod += S[k][i] * D[k] * S[k][j]
            S[i][j] = (A[i][j] - sum_prod) / (S[i][i] * D[i])

    return S, D


def solve_general_cholesky(A, b):
    n = len(A)
    S, D = general_cholesky_decomposition(A)

    y = [0.0] * n
    for i in range(n):
        sum_prod = 0.0
        for j in range(i):
            sum_prod += S[j][i] * D[j] * y[j]
        y[i] = (b[i] - sum_prod) / (S[i][i] * D[i])

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_prod = 0.0
        for j in range(i + 1, n):
            sum_prod += S[i][j] * x[j]
        x[i] = (y[i] - sum_prod) / S[i][i]

    return x, S, D


def main():
    n = int(input("Введите размерность матрицы n: "))

    print(f"Введите матрицу A размером {n}x{n} (по строкам):")
    A = []
    for i in range(n):
        row = list(map(float, input().split()))
        A.append(row)

    print(f"Введите вектор b размером {n}:")
    b = list(map(float, input().split()))

    try:
        x, S, D = solve_general_cholesky(A, b)

        print("\nМатрица S:")
        for i in range(n):
            print([f"{S[i][j]:10.6f}" for j in range(n)])

        print("\nМатрица D:")
        for i in range(n):
            row = [0.0] * n
            row[i] = D[i]
            print([f"{row[j]:10.1f}" for j in range(n)])

        print("\nРешение x:")
        print([f"{val:10.6f}" for val in x])

    except ValueError as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
