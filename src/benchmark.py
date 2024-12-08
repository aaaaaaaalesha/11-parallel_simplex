import time
from typing import Any, Iterable

import numpy as np
from prettytable import PrettyTable

from src.simplex.simplex_problem import SimplexProblem


def benchmark_lp_solver(problem_size_iterable: Iterable[int]) -> PrettyTable:
    # Создание таблицы для результатов
    report_table = PrettyTable(("Размерность задачи (число переменных / ограничений)",
                                "CPU Time (ms)", "GPU Time (ms)"))
    for problem_size in problem_size_iterable:
        # Генерация входных данных
        lp_input = _generate_lp_input(problem_size, problem_size)
        # Решение задачи без GPU.
        problem_cpu = SimplexProblem.from_constraints(**lp_input,
                                                      verbose=False)
        start_time = time.time()
        sol_cpu = problem_cpu.solve()
        end_time = time.time()
        cpu_time = (end_time - start_time) * 1_000  # ms

        # Решение задачи с GPU.
        problem_gpu = SimplexProblem.from_constraints(**lp_input,
                                                      use_gpu=True, verbose=False)
        start_time = time.time()
        sol_gpu = problem_gpu.solve()
        end_time = time.time()
        gpu_time = (end_time - start_time) * 1_000  # ms

        assert sol_cpu[1] == sol_gpu[1]
        # Добавление результатов в таблицу.
        report_table.add_row([problem_size, f"{cpu_time:.3f}", f"{gpu_time:.3f}"])

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
