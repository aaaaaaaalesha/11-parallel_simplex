"""Реализация симплекс-таблицы."""

import logging
import warnings

import numpy as np
from prettytable import PrettyTable

from .exceptions import SimplexProblemException

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SimplexTable:
    """Класс симплекс-таблицы."""

    def __init__(
            self,
            obj_func_coffs: np.array,
            constraint_system_lhs: np.array,
            constraint_system_rhs: np.array,
    ):
        """
        :param obj_func_coffs: Коэффициенты ЦФ.
        :param constraint_system_lhs: Левая часть системы ограничений.
        :param constraint_system_rhs: Правая часть системы ограничений.
        """
        var_count = len(obj_func_coffs)
        constraint_count = constraint_system_lhs.shape[0]

        # Заполнение верхнего хедера.
        self.top_row_ = ["  ", "Si0"] + [f"x{i + 1}" for i in range(var_count)]
        # Заполнение левого хедера.
        self.left_column_ = [f"x{var_count + i + 1}" for i in range(constraint_count)] + ["F "]

        self.main_table_ = np.zeros((constraint_count + 1, var_count + 1))
        # Заполняем столбец Si0.
        for i in range(constraint_count):
            self.main_table_[i][0] = constraint_system_rhs[i]
        # Заполняем строку F.
        for j in range(var_count):
            self.main_table_[constraint_count][j + 1] = -obj_func_coffs[j]

        # Заполняем А.
        for i in range(constraint_count):
            for j in range(var_count):
                self.main_table_[i][j + 1] = constraint_system_lhs[i][j]

    def __str__(self):
        table = PrettyTable(self.top_row_, float_format=".4")
        for i in range(self.main_table_.shape[0]):
            table.add_row([self.left_column_[i], *self.main_table_[i]])

        return str(table)

    def is_find_ref_solution(self) -> bool:
        """
        Проверяет, найдено ли опорное решение по свободным в симплекс-таблице.
        :return: True - опорное решение уже найдено, иначе - пока не является опорным.
        """
        return all(self.main_table_[i][0] >= 0 for i in range(self.main_table_.shape[0] - 1))

    def search_ref_solution(self) -> None:
        """
        Функция производит одну итерацию поиска опорного решения.
        """
        res_row = None
        for i in range(self.main_table_.shape[0] - 1):
            if self.main_table_[i][0] < 0:
                res_row = i
                break

        # Если найден отрицательный элемент в столбце свободных членов,
        # то ищем первый отрицательный в строке с ней.
        res_col = None
        if res_row is not None:
            for j in range(1, self.main_table_.shape[1]):
                if self.main_table_[res_row, j] < 0:
                    res_col = j
                    break

        # Если найден разрешающий столбец, то находим в нём разрешающий элемент.
        if res_col is None:
            exc_msg = (
                "Задача не имеет допустимых решений! "
                "При нахождении опорного решения не нашлось "
                "отрицательного элемента в строке с отрицательным свободным членом."
            )
            raise SimplexProblemException(exc_msg)

        # Ищем минимальное положительное отношение Si0 / x[res_col].
        minimum = None
        ind = -1
        for i in range(self.main_table_.shape[0] - 1):
            # Ищем минимальное отношение -- разрешающую строку.
            curr = self.main_table_[i][res_col]
            s_i0 = self.main_table_[i][0]
            if curr == 0:
                continue

            if (s_i0 / curr) > 0 and (minimum is None or (s_i0 / curr) < minimum):
                minimum = s_i0 / curr
                ind = i

        if minimum is None:
            exc_msg = (
                "Решения не существует! "
                "При нахождении опорного решения не нашлось минимального "
                "положительного отношения."
            )
            raise SimplexProblemException(exc_msg)

        res_row = ind
        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row][res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col + 1])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)

    def is_find_opt_solution(self) -> bool:
        """
        Проверяет, найдено ли оптимальное решение по коэффициентам ЦФ в симплекс-таблице.
        :return: True - оптимальное решение уже найдено, иначе - пока не оптимально.
        """
        # Если положительных не нашлось, то оптимальное решение уже найдено.
        return all(self.main_table_[self.main_table_.shape[0] - 1][i] <= 0 for i in range(1, self.main_table_.shape[1]))

    def optimize_ref_solution(self) -> None:
        """
        Производит одну итерацию поиска оптимального решения на основе
        уже полученного опорного решения.
        """
        res_col = None
        ind_f = self.main_table_.shape[0] - 1

        # В строке F ищем первый положительный.
        for j in range(1, self.main_table_.shape[1]):
            curr = self.main_table_[ind_f][j]
            if curr > 0:
                res_col = j
                break

        minimum, res_row = None, None
        # Идём по всем, кроме ЦФ ищём минимальное отношение.
        for i in range(self.main_table_.shape[0] - 1):
            # Ищем минимальное отношение - разрешающую строку.
            curr = self.main_table_[i][res_col]
            s_i0 = self.main_table_[i][0]
            if curr < 0:
                continue

            if (s_i0 / curr) >= 0 and (minimum is None or (s_i0 / curr) < minimum):
                minimum = s_i0 / curr
                res_row = i

        if res_row is None:
            exc_msg = "Функция не ограничена! Оптимального решения не существует."
            raise SimplexProblemException(exc_msg)

        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row][res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col + 1])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)

    def recalculate_table(self, res_row: int, res_col: int, res_element):
        """
        По заданным разрешающим строке, столбцу и элементу производит пересчёт
        симплекс-таблицы методом Жордановых исключений.
        :param res_row: Индекс разрешающей строки.
        :param res_col: Индекс разрешающего столбца.
        :param res_element: Разрешающий элемент.
        """
        # Пересчёт таблицы (далее оставим всё, за исключением разрешающих строки/столбца/элемента).
        recalculated_table = self.main_table_ - (
                self.main_table_[:, res_col][:, np.newaxis] * self.main_table_[res_row, :]
        ) / res_element

        # Пересчёт разрешающей строки.
        for j in range(self.main_table_.shape[1]):
            if j != res_col:
                recalculated_table[res_row][j] = self.main_table_[res_row][j] / res_element

        # Пересчёт разрешающего столбца.
        for i in range(self.main_table_.shape[0]):
            if i != res_row:
                recalculated_table[i][res_col] = -(self.main_table_[i][res_col] / res_element)

        # Пересчёт разрешающего элемента.
        recalculated_table[res_row][res_col] = 1 / res_element

        self.main_table_ = recalculated_table
        self.swap_headers(res_row, res_col)
        _logger.info("%s", self)

    def swap_headers(self, res_row: int, res_col: int) -> None:
        """
        Меняет переменные в строке и столбце местами.
        :param res_row: Индекс разрешающей строки.
        :param res_col: Индекс разрешающего столбца.
        """
        self.top_row_[res_col + 1], self.left_column_[res_row] = (
            self.left_column_[res_row],
            self.top_row_[res_col + 1],
        )
