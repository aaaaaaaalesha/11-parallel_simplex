"""Реализация симплекс-таблицы для Numba."""

import logging
import warnings

import numpy as np
from numba import cuda, float32

from src.simplex.exceptions import SimplexProblemException
from src.simplex.simplex_table.base import BaseSimplexTable

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@cuda.jit
def gpu_recalculate_table(main_table, res_row, res_col, res_element):
    row, col = cuda.grid(2)
    if row < main_table.shape[0] and col < main_table.shape[1]:
        if row == res_row:
            main_table[row, col] /= res_element
        elif col == res_col:
            main_table[row, col] = - (main_table[row, col] / res_element)
        else:
            main_table[row, col] -= (main_table[res_row, col] * main_table[row, res_col]) / res_element

    if row == res_row and col == res_col:
        main_table[row, col] = 1 / res_element


class NumbaSimplexTable(BaseSimplexTable):
    """Класс симплекс-таблицы, использующий CUDA для организации вычислений на GPU."""

    def __init__(self, obj_func_coffs: np.array, constraint_system_lhs: np.array, constraint_system_rhs: np.array):
        super().__init__(obj_func_coffs, constraint_system_lhs, constraint_system_rhs)
        self.device_main_table = cuda.to_device(self.main_table_)

    def recalculate_table(self, res_row: int, res_col: int, res_element: float32):
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.main_table_.shape[0] + (threads_per_block[0] - 1)) // threads_per_block[0]
        blocks_per_grid_y = (self.main_table_.shape[1] + (threads_per_block[1] - 1)) // threads_per_block[1]

        # Запуск ядра CUDA для пересчета таблицы
        gpu_recalculate_table[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
            self.device_main_table, res_row, res_col, res_element
        )

        # Копируем результат обратно на хост
        self.main_table_ = self.device_main_table.copy_to_host()
        self.swap_headers(res_row, res_col)
        _logger.info("%s", self)

    def search_ref_solution(self) -> None:
        """Функция производит одну итерацию поиска опорного решения на GPU."""
        negative_rows = np.where(self.main_table_[:-1, 0] < 0)[0]

        if negative_rows.size == 0:
            raise SimplexProblemException("Задача не имеет допустимых решений!")

        res_row = negative_rows[0]
        negative_columns = np.where(self.main_table_[res_row, 1:] < 0)[0]

        if negative_columns.size == 0:
            raise SimplexProblemException("Задача не имеет допустимых решений!")

        res_col = negative_columns[0] + 1
        ratios = self.main_table_[:-1, 0] / self.main_table_[:-1, res_col]
        positive_ratios = ratios[ratios > 0]

        if positive_ratios.size == 0:
            raise SimplexProblemException("Решения не существует!")

        ind = np.argmin(positive_ratios)
        res_row = negative_rows[ind]
        res_element = self.main_table_[res_row, res_col]

        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        self.recalculate_table(res_row, res_col, res_element)

    def optimize_ref_solution(self) -> None:
        """Производит одну итерацию поиска оптимального решения на GPU."""
        ind_f = self.main_table_.shape[0] - 1
        positive_columns = np.where(self.main_table_[ind_f, 1:] > 0)[0]

        if positive_columns.size == 0:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        res_col = positive_columns[0] + 1
        s_i0 = self.main_table_[:-1, 0]
        curr = self.main_table_[:-1, res_col]
        valid_rows = curr >= 0
        ratios = np.where(valid_rows, s_i0 / curr, np.inf)

        if np.all(np.isinf(ratios)):
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        res_row = np.argmin(ratios)

        if not valid_rows[res_row]:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        res_element = self.main_table_[res_row, res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        self.recalculate_table(res_row, res_col, res_element)
