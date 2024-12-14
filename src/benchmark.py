import gc
from typing import Any, Iterable

import numpy as np
from prettytable import PrettyTable

from src.simplex.simplex_problem import SimplexProblem


def benchmark_lp_solver(problem_size_iterable: Iterable[int]) -> PrettyTable:
    # Создание таблицы для результатов
    report_table = PrettyTable(("Размерность задачи (число переменных / ограничений)",
                                "CPU Time (ms)", "GPU Time (ms)",
                                "Прирост производительности"))
    for problem_size in problem_size_iterable:
        print(f"Processing problem with size: {problem_size}")
        # Генерация входных данных
        lp_input = _generate_lp_input(problem_size, problem_size)
        # Решение задачи на CPU.
        problem_cpu = SimplexProblem.from_constraints(**lp_input, verbose=False)
        sol_cpu, timer = problem_cpu.solve(timer=True)
        cpu_time = timer * 1_000  # ms
        print(f"  CPU completed: {cpu_time:.3f} ms")
        del problem_cpu
        gc.collect()

        # Решение задачи с GPU.
        problem_gpu = SimplexProblem.from_constraints(**lp_input, use_gpu="cupy", verbose=False)
        sol_gpu, timer = problem_gpu.solve(timer=True)
        gpu_time = timer * 1_000  # ms
        print(f"  GPU completed: {gpu_time:.3f} ms")
        del problem_gpu
        gc.collect()

        assert sol_cpu[1] == sol_gpu[1]
        print(f"  CPU and GPU have found equivalent solution: f = {sol_cpu[1]:.3f}")
        del sol_cpu, sol_gpu
        gc.collect()
        # Добавление результатов в таблицу.
        report_table.add_row([problem_size, f"{cpu_time:.3f}",
                              f"{gpu_time:.3f}", f"{cpu_time / gpu_time:.2f}"])  # f"{0:.3f}"])


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
