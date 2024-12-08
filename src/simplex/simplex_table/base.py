"""Базовая реализация симплекс-таблицы."""

import logging
import warnings

import numpy as np
from numpy import signedinteger, float32
from prettytable import PrettyTable

from src.simplex.exceptions import SimplexProblemException

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class BaseSimplexTable:
    """Базовый класс симплекс-таблицы."""

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

        self.main_table_: np.array = np.zeros((constraint_count + 1, var_count + 1))
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
        return all(self.main_table_[i][0] >= 0
                   for i in range(self.main_table_.shape[0] - 1))

    def search_ref_solution(self) -> None:
        """Функция производит одну итерацию поиска опорного решения."""
        # Находим строки с отрицательными свободными членами
        negative_rows = np.where(self.main_table_[:-1, 0] < 0)[0]

        if negative_rows.size == 0:
            raise SimplexProblemException("Задача не имеет допустимых решений!")

        # Если найден отрицательный элемент в столбце свободных членов,
        # то ищем первый отрицательный в строке с ней.
        res_row = negative_rows[0]
        # Находим первый отрицательный элемент в строке с отрицательным свободным членом.
        negative_columns = np.where(self.main_table_[res_row, 1:] < 0)[0]

        if negative_columns.size == 0:
            raise SimplexProblemException(
                "Задача не имеет допустимых решений! "
                "При нахождении опорного решения не нашлось "
                "отрицательного элемента в строке с отрицательным свободным членом."
            )
        # Если найден разрешающий столбец, то находим в нём разрешающий элемент.
        # `+1`, чтобы учесть смещение из-за первого столбца
        res_col: signedinteger = negative_columns[0] + 1

        # Ищем минимальное положительное отношение Si0 / x[res_col]
        ratios = self.main_table_[:-1, 0] / self.main_table_[:-1, res_col]
        positive_ratios = ratios[ratios > 0]

        if positive_ratios.size == 0:
            raise SimplexProblemException(
                "Решения не существует! При нахождении опорного решения не нашлось минимального "
                "положительного отношения."
            )

        # Находим индекс разрешающей строки среди строк с отрицательными свободными членами
        ind = np.argmin(positive_ratios)
        # Получаем индекс разрешающей строки.
        res_row: signedinteger = negative_rows[ind]

        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row, res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)

    def is_find_opt_solution(self) -> bool:
        """
        Проверяет, найдено ли оптимальное решение по коэффициентам ЦФ в симплекс-таблице.
        :return: True - оптимальное решение уже найдено, иначе - пока не оптимально.
        """
        # Если положительных не нашлось, то оптимальное решение уже найдено.
        return all(self.main_table_[self.main_table_.shape[0] - 1][i] <= 0
                   for i in range(1, self.main_table_.shape[1]))

    def optimize_ref_solution(self) -> None:
        """
        Производит одну итерацию поиска оптимального решения на основе
        уже полученного опорного решения.
        """
        ind_f: int = self.main_table_.shape[0] - 1
        # Находим первый положительный элемент в строке F.
        positive_columns = np.where(self.main_table_[ind_f, 1:] > 0)[0]

        if positive_columns.size == 0:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # `+1`, чтобы учесть смещение из-за первого столбца.
        res_col = positive_columns[0] + 1

        # Ищем минимальное отношение `Si0 / x[res_col]` (Идём по всем, кроме ЦФ ищём минимальное отношение).
        s_i0 = self.main_table_[:-1, 0]
        curr = self.main_table_[:-1, res_col]
        # Находим положительные элементы в текущем столбце.
        valid_rows = curr >= 0
        # Заменяем недопустимые отношения на бесконечность.
        ratios = np.where(valid_rows, s_i0 / curr, np.inf)

        # Находим минимальное положительное отношение.
        if np.all(np.isinf(ratios)):
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # Находим индекс разрешающей строки.
        res_row = np.argmin(ratios)

        # Проверяем, что `res_row` соответствует индексу в исходной таблице.
        if not valid_rows[res_row]:
            raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

        # Разрешающий элемент найден.
        res_element = self.main_table_[res_row, res_col]
        _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
        _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

        # Пересчёт симплекс-таблицы.
        self.recalculate_table(res_row, res_col, res_element)

    def recalculate_table(self, res_row: signedinteger, res_col: signedinteger, res_element: float32):
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
        self.top_row_[res_col + 1], self.left_column_[res_row] = self.left_column_[res_row], self.top_row_[res_col + 1]
