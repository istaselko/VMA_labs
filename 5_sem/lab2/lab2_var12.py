import numpy as np
import math

def print_matrix(matrix, title=""):
    """
    Выводит матрицу в форматированном виде.
    """
    if title:
        print(title)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    
    # Определяем формат вывода
    for row in matrix:
        print("  ".join("{:10.5f}".format(x) for x in row))
    print()

def generate_symmetric_matrix(n):
    """
    Генерирует случайную симметричную матрицу размера n x n.
    Симметричность обязательна для метода вращений (Якоби).
    """
    A = np.random.rand(n, n) * 10
    # Делаем симметричной: A = (A + A^T) / 2
    A_sym = (A + A.T) / 2
    return A_sym

def get_max_off_diagonal(A):
    """
    Находит максимальный по модулю внедиагональный элемент.
    Возвращает значение и индексы (max_val, p, q).
    """
    n = A.shape[0]
    max_val = 0.0
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return max_val, p, q

def jacobi_rotation_method(A, tol=1e-10, max_iter=10000):
    """
    Метод вращений (Якоби) для решения полной проблемы собственных значений.
    Возвращает: собственные числа (вектор) и собственные векторы (матрица).
    """
    n = A.shape[0]
    # Копируем матрицу, так как будем ее менять
    A_curr = A.copy()
    # Инициализируем матрицу собственных векторов как единичную
    V = np.eye(n)
    
    iterations = 0
    while True:
        # 1. Находим максимальный внедиагональный элемент
        max_val, p, q = get_max_off_diagonal(A_curr)
        
        # 2. Критерий остановки
        if max_val < tol or iterations >= max_iter:
            break
            
        iterations += 1
        
        # 3. Вычисление угла поворота phi
        # Используем формулы: tan(2phi) = 2*a_pq / (a_pp - a_qq)
        if A_curr[p, p] == A_curr[q, q]:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(2 * A_curr[p, q] / (A_curr[p, p] - A_curr[q, q]))
            
        c = math.cos(phi)
        s = math.sin(phi)
        
        # 4. Пересчет элементов матрицы (вращение)
        # Создаем временную копию для корректного пересчета
        A_next = A_curr.copy()
        
        # Элементы в узлах вращения
        A_next[p, p] = c**2 * A_curr[p, p] + 2 * c * s * A_curr[p, q] + s**2 * A_curr[q, q]
        A_next[q, q] = s**2 * A_curr[p, p] - 2 * c * s * A_curr[p, q] + c**2 * A_curr[q, q]
        A_next[p, q] = 0 # Теоретически 0, принудительно зануляем
        A_next[q, p] = 0
        
        # Остальные элементы в строках/столбцах p и q
        for i in range(n):
            if i != p and i != q:
                A_next[i, p] = c * A_curr[i, p] + s * A_curr[i, q]
                A_next[p, i] = A_next[i, p] # Симметрия
                
                A_next[i, q] = -s * A_curr[i, p] + c * A_curr[i, q]
                A_next[q, i] = A_next[i, q] # Симметрия
        
        A_curr = A_next
        
        # 5. Накопление собственных векторов
        # V_new = V_old * G, где G - матрица вращения.
        # Это меняет только p-й и q-й столбцы матрицы V
        V_next = V.copy()
        for i in range(n):
            V_next[i, p] = c * V[i, p] + s * V[i, q]
            V_next[i, q] = -s * V[i, p] + c * V[i, q]
        V = V_next

    eigenvalues = np.diag(A_curr)
    return eigenvalues, V, iterations

def power_iteration_method(A, tol=1e-10, max_iter=1000):
    """
    Степенной метод для нахождения максимального по модулю собственного значения.
    """
    n = A.shape[0]
    # Случайный начальный вектор
    x = np.random.rand(n)
    x = x / np.linalg.norm(x) # Нормировка
    
    lam_prev = 0
    
    for k in range(max_iter):
        # x_next = A * x
        x_next = np.dot(A, x)
        
        # Оценка собственного числа (через отношение Релея для симметричных матриц
        # или просто скалярное произведение, если вектор нормирован)
        # lam = (Ax, x) / (x, x) -> так как (x,x)=1, то lam = (x_next, x)
        lam = np.dot(x_next, x)
        
        # Нормировка вектора
        norm = np.linalg.norm(x_next)
        if norm == 0:
            break
        x_next = x_next / norm
        
        # Проверка сходимости
        if abs(lam - lam_prev) < tol:
            return lam, x_next, k + 1
            
        lam_prev = lam
        x = x_next
        
    return lam_prev, x, max_iter

def main():
    while True:
        try:
            val = input("Введите размерность матрицы n (>=10) [Enter для n=10]: ")
            if not val:
                n = 10
            else:
                n = int(val)
            if n < 2:
                print("n должно быть >= 2")
                continue
            break
        except ValueError:
            print("Введите целое число.")
            
    print(f"\n--- Генерация симметричной матрицы {n}x{n} ---")
    A = generate_symmetric_matrix(n)
    if n <= 10:
        print_matrix(A, "Матрица A:")
    else:
        print("Матрица сгенерирована (вывод скрыт из-за размера).")

    # ==========================================
    # 1. Метод вращений (Якоби)
    # ==========================================
    print("\n" + "="*50)
    print("ЗАДАНИЕ 2: Решение полной проблемы (Метод вращений)")
    print("="*50)
    
    eig_vals_jacobi, eig_vecs_jacobi, iters_jacobi = jacobi_rotation_method(A)
    
    print(f"Количество итераций: {iters_jacobi}")
    print("\nНайденные собственные числа (Якоби):")
    # Сортируем для удобства восприятия
    idx = np.argsort(eig_vals_jacobi)[::-1]
    sorted_eig_vals = eig_vals_jacobi[idx]
    sorted_eig_vecs = eig_vecs_jacobi[:, idx]
    
    print(sorted_eig_vals)
    
    # Проверка для максимального по модулю
    max_idx = np.argmax(np.abs(sorted_eig_vals))
    lambda_max_jacobi = sorted_eig_vals[max_idx]
    vec_max_jacobi = sorted_eig_vecs[:, max_idx]
    
    # Проверка ||Ax - lambda*x|| для Якоби (для макс. числа)
    residual_jacobi = np.linalg.norm(np.dot(A, vec_max_jacobi) - lambda_max_jacobi * vec_max_jacobi)
    print(f"\nПроверка для макс. с.ч. (Якоби) ||Ax - lambda*x||: {residual_jacobi:.5e}")

    # ==========================================
    # 2. Степенной метод
    # ==========================================
    print("\n" + "="*50)
    print("ЗАДАНИЕ 3: Поиск наибольшего с.з. (Степенной метод)")
    print("="*50)
    
    lam_pow, vec_pow, iters_pow = power_iteration_method(A)
    
    print(f"Количество итераций: {iters_pow}")
    print(f"Максимальное по модулю с.з. (Power): {lam_pow:.8f}")
    
    # Сравнение знаков векторов (вектор может быть направлен в другую сторону)
    # Если скалярное произведение векторов отрицательное, инвертируем один для сравнения
    if np.dot(vec_pow, vec_max_jacobi) < 0:
        vec_pow = -vec_pow
        
    residual_pow = np.linalg.norm(np.dot(A, vec_pow) - lam_pow * vec_pow)
    print(f"Проверка (Power) ||Ax - lambda*x||: {residual_pow:.5e}")

    # ==========================================
    # 3. Сравнение результатов
    # ==========================================
    print("\n" + "="*50)
    print("СРАВНЕНИЕ И ВЫВОДЫ")
    print("="*50)
    
    diff_val = abs(abs(lambda_max_jacobi) - abs(lam_pow))
    print(f"Разница между с.з. Якоби и Степенного метода: {diff_val:.5e}")
    
    # Используем встроенную функцию для "истинных" значений
    true_vals = np.linalg.eigvalsh(A)
    max_true = max(abs(true_vals))
    print(f"Библиотечное значение (numpy.linalg.eigvalsh): {max_true:.8f}")

if __name__ == "__main__":
    main()