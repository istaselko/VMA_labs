def print_matrix(mat):
    for row in mat:
        print("  ".join(f"{x:10.4f}" for x in row))
    print()

def gauss_solve(matrix, b):
    n = len(matrix)
    # Создаём расширенную матрицу
    aug = [matrix[i] + [b[i]] for i in range(n)]
    
    # Прямой ход
    for i in range(n):
        # Нормируем ведущий элемент
        pivot = aug[i][i]
        if pivot == 0:
            # Ищем строку с ненулевым элементом для обмена
            for j in range(i+1, n):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
            else:
                raise ValueError("Система не имеет уникального решения")
        for k in range(i, n+1):
            aug[i][k] /= pivot
        
        # Обнуляем элементы ниже
        for j in range(i+1, n):
            factor = aug[j][i]
            for k in range(i, n+1):
                aug[j][k] -= factor * aug[i][k]
    
    # Обратный ход
    x = [0]*n
    for i in range(n-1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i+1, n):
            x[i] -= aug[i][j] * x[j]
    return x

def gauss_inverse(matrix):
    n = len(matrix)
    # Начинаем с единичной матрицы
    inverse = [[float(i==j) for j in range(n)] for i in range(n)]
    
    # Копируем матрицу, чтобы не изменять оригинал
    mat = [row[:] for row in matrix]
    
    # Прямой ход
    for i in range(n):
        pivot = mat[i][i]
        if pivot == 0:
            # Меняем строки
            for j in range(i+1, n):
                if mat[j][i] != 0:
                    mat[i], mat[j] = mat[j], mat[i]
                    inverse[i], inverse[j] = inverse[j], inverse[i]
                    pivot = mat[i][i]
                    break
            else:
                raise ValueError("Матрица необратима")
        for k in range(n):
            mat[i][k] /= pivot
            inverse[i][k] /= pivot
        for j in range(n):
            if j != i:
                factor = mat[j][i]
                for k in range(n):
                    mat[j][k] -= factor * mat[i][k]
                    inverse[j][k] -= factor * inverse[i][k]
    return inverse

# Ввод данных
n = int(input("Введите размер матрицы: "))
matrix = []
print("Введите элементы матрицы построчно через пробел:")
for i in range(n):
    row = list(map(float, input(f"Строка {i+1}: ").split()))
    matrix.append(row)

print("\nВыберите действие:")
print("1. Решить систему методом Гаусса")
print("2. Найти обратную матрицу методом Гаусса")
choice = input("Ваш выбор (1 или 2): ")

if choice == "1":
    b = list(map(float, input("Введите столбец свободных членов через пробел: ").split()))
    solution = gauss_solve(matrix, b)
    print("Решение системы:")
    for i, val in enumerate(solution):
        print(f"x{i+1} = {val:.4f}")
elif choice == "2":
    inv = gauss_inverse(matrix)
    print("Обратная матрица:")
    print_matrix(inv)
else:
    print("Неверный выбор")
