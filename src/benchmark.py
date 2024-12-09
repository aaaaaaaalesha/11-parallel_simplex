import time
from typing import Any, Iterable

import numpy as np
from prettytable import PrettyTable

from src.simplex.simplex_problem import SimplexProblem


def benchmark_lp_solver(problem_size_iterable: Iterable[int]) -> PrettyTable:
    # Создание таблицы для результатов
    report_table = PrettyTable(("Размерность задачи (число переменных / ограничений)",
                                "CPU Time (ms)", "GPU Time (ms)",
                                "Прирост производительности (?)"))  # "Numba GPU Time (ms)"))
    for problem_size in problem_size_iterable:
        # Генерация входных данных
        lp_input = _generate_lp_input(problem_size, problem_size)
        # Решение задачи без GPU.
        problem_cpu = SimplexProblem.from_constraints(**lp_input, verbose=False)
        sol_cpu, timer = problem_cpu.solve(timer=True)
        cpu_time = timer * 1_000  # ms

        # Решение задачи с GPU (CuPy).
        problem_gpu = SimplexProblem.from_constraints(**lp_input, use_gpu="cupy", verbose=False)
        sol_cupy, timer = problem_gpu.solve(timer=True)
        cupy_time = timer * 1_000  # ms

        assert sol_cpu[1] == sol_cupy[1]
        # Добавление результатов в таблицу.
        report_table.add_row([problem_size, f"{cpu_time:.3f}",
                              f"{cupy_time:.3f}", f"{cpu_time / cupy_time:.2f}"])  # f"{0:.3f}"])

    return report_table


def _generate_lp_input(num_vars: int, num_constraints: int) -> dict[str, Any]:
    # Генерация коэффициентов целевой функции.
    obj_func_coffs = np.random.rand(num_vars).tolist()

    # Генерация системы ограничений.
    constraint_system_lhs = np.random.rand(num_constraints, num_vars).tolist()
    constraint_system_rhs = np.random.rand(num_constraints).tolist()

    # Формирование JSON-структуры
    lp_input = {
        "obj_func_coffs": obj_func_coffs,
        "constraint_system_lhs": constraint_system_lhs,
        "constraint_system_rhs": constraint_system_rhs,
        "func_direction": "max"
    }

    return lp_input
