"""Реализация симплекс-таблицы под CuPy для вычислений на GPU."""

import logging
import warnings

import cupy as cp
import numpy as np

from src.simplex.exceptions import SimplexProblemException
from src.simplex.simplex_table.base import BaseSimplexTable

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class CupySimplexTable(BaseSimplexTable):
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

    def search_ref_solution(self) -> None:
        """Функция производит одну итерацию поиска опорного решения."""

        # Находим строки с отрицательными свободными членами
        negative_rows = cp.where(self.main_table_[:-1, 0] < 0)[0]

        if negative_rows.size == 0:
            raise SimplexProblemException("Задача не имеет допустимых решений!")

        # Если найден отрицательный элемент в столбце свободных членов,
        # то ищем первый отрицательный в строке с ней.
        res_row = negative_rows[0]
        # Находим первый отрицательный элемент в строке с отрицательным свободным членом.
        negative_columns = cp.where(self.main_table_[res_row, 1:] < 0)[0]

        if negative_columns.size == 0:
            raise SimplexProblemException(
                "Задача не имеет допустимых решений! "
                "При нахождении опорного решения не нашлось "
                "отрицательного элемента в строке с отрицательным свободным членом."
            )
        # Если найден разрешающий столбец, то находим в нём разрешающий элемент.
        # `+1`, чтобы учесть смещение из-за первого столбца
        res_col: cp.signedinteger = negative_columns[0] + 1

        # Ищем минимальное положительное отношение Si0 / x[res_col]
        ratios = self.main_table_[:-1, 0] / self.main_table_[:-1, res_col]
        positive_ratios = ratios[ratios > 0]

        if positive_ratios.size == 0:
            raise SimplexProblemException(
                "Решения не существует! При нахождении опорного решения не нашлось минимального "
                "положительного отношения."
            )

        # Находим индекс разрешающей строки среди строк с отрицательными свободными членами
        ind = cp.argmin(positive_ratios)
        # Получаем индекс разрешающей строки.
        res_row: cp.signedinteger = negative_rows[ind]

        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row, res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)

    def optimize_ref_solution(self) -> None:
        """
        Производит одну итерацию поиска оптимального решения на основе
        уже полученного опорного решения.
        """
        ind_f: int = self.main_table_.shape[0] - 1
        # Находим первый положительный элемент в строке F.
        positive_columns = cp.where(self.main_table_[ind_f, 1:] > 0)[0]

        if positive_columns.size == 0:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # `+1`, чтобы учесть смещение из-за первого столбца.
        res_col = int((positive_columns[0] + 1).item())

        # Ищем минимальное отношение `Si0 / x[res_col]` (Идём по всем, кроме ЦФ ищём минимальное отношение).
        s_i0 = self.main_table_[:-1, 0]
        curr = self.main_table_[:-1, res_col]
        # Находим положительные элементы в текущем столбце.
        valid_rows = curr >= 0
        # Заменяем недопустимые отношения на бесконечность.
        ratios = cp.where(valid_rows, s_i0 / curr, cp.inf)

        # Находим минимальное положительное отношение.
        if cp.all(cp.isinf(ratios)):
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # Находим индекс разрешающей строки.
        res_row = int(cp.argmin(ratios).item())

        # Проверяем, что `res_row` соответствует индексу в исходной таблице.
        if not valid_rows[res_row]:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row, res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)
