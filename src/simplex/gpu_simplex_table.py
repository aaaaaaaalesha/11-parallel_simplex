"""Реализация симплекс-таблицы."""

import logging
import warnings

import cupy as cp
import numpy as np

from .simplex_table import SimplexTable

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GPUSimplexTable(SimplexTable):
    """Класс симплекс-таблицы, использующий CUDA для организации вычислений на GPU."""

    def __init__(self, obj_func_coffs: np.ndarray, constraint_system_lhs: np.ndarray,
                 constraint_system_rhs: np.ndarray):
        """
        Инициализация симплекс-таблицы с использованием CuPy для вычислений на GPU.
        :param obj_func_coffs: Коэффициенты ЦФ.
        :param constraint_system_lhs: Левая часть системы ограничений.
        :param constraint_system_rhs: Правая часть системы ограничений.
        """
        super().__init__(obj_func_coffs, constraint_system_lhs, constraint_system_rhs)
        # Переносим основную таблицу на GPU для последующих вычислений.
        self.main_table_: cp.array = cp.asarray(self.main_table_)

    def is_find_ref_solution(self) -> bool:
        """
        Проверяет, найдено ли опорное решение по свободным в симплекс-таблице.
        :return: True - опорное решение уже найдено, иначе - пока не является опорным.
        """
        return bool(cp.all(self.main_table_[:-1, 0] >= 0).item())

    def is_find_opt_solution(self) -> bool:
        """
        Проверяет, найдено ли оптимальное решение по коэффициентам ЦФ в симплекс-таблице.
        :return: True - оптимальное решение уже найдено, иначе - пока не оптимально.
        """
        return bool(cp.all(self.main_table_[-1, 1:] <= 0).item())
