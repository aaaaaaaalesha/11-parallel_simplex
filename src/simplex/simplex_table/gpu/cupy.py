"""Реализация симплекс-таблицы под CuPy для вычислений на GPU."""

import logging
import warnings

import cupy as cp
import numpy as np

from src.simplex.exceptions import SimplexProblemException
from src.simplex.simplex_table.base import BaseSimplexTable
from src.simplex.types import Extremum

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class CupySimplexTable(BaseSimplexTable):
    """Класс симплекс-таблицы, использующий CUDA для организации вычислений на GPU."""

    def find_optimal_solution(self, extremum: Extremum = "max", verbose=True) -> None:
        # Переносим основную таблицу на GPU для последующих вычислений.
        main_table: cp.array = cp.asarray(self.main_table_)
        # Этап 1: Поиск опорного решения.
        _logger.info("Поиск опорного решения: \nИсходная симплекс-таблица:\n%s", self)
        while not bool(cp.all(main_table[:-1, 0] >= 0).item()):
            # Итерация поиска опорного решения.
            # Находим строки с отрицательными свободными членами
            negative_rows = cp.where(main_table[:-1, 0] < 0)[0]

            if negative_rows.size == 0:
                raise SimplexProblemException("Задача не имеет допустимых решений!")

            # Если найден отрицательный элемент в столбце свободных членов,
            # то ищем первый отрицательный в строке с ней.
            res_row = negative_rows[0]
            # Находим первый отрицательный элемент в строке с отрицательным свободным членом.
            negative_columns = cp.where(main_table[res_row, 1:] < 0)[0]

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
            ratios = main_table[:-1, 0] / main_table[:-1, res_col]
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
            res_element = main_table[res_row, res_col]
            _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
            _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

            # Пересчёт симплекс-таблицы.
            recalculated_table = main_table - (
                    main_table[:, res_col][:, np.newaxis] * main_table[res_row, :]
            ) / res_element

            # Пересчёт разрешающей строки.
            for j in range(main_table.shape[1]):
                if j != res_col:
                    recalculated_table[res_row][j] = main_table[res_row][j] / res_element

            # Пересчёт разрешающего столбца.
            for i in range(main_table.shape[0]):
                if i != res_row:
                    recalculated_table[i][res_col] = -(main_table[i][res_col] / res_element)

            # Пересчёт разрешающего элемента.
            recalculated_table[res_row][res_col] = 1 / res_element

            main_table = recalculated_table
            self.swap_headers(res_row, res_col)
            _logger.info("%s", self)

        _logger.info("Опорное решение найдено!")
        if verbose:
            self.output_solution()

        # Этап 2: Поиск оптимального решения.
        _logger.info("Поиск оптимального решения:")
        while not bool(cp.all(main_table[-1, 1:] <= 0).item()):
            # Производит одну итерацию поиска оптимального решения на основе уже полученного опорного решения.
            ind_f: int = main_table.shape[0] - 1
            # Находим первый положительный элемент в строке F.
            positive_columns = cp.where(main_table[ind_f, 1:] > 0)[0]

            if positive_columns.size == 0:
                raise SimplexProblemException("Функция не ограничена! Оптимального решения не существует.")

            # `+1`, чтобы учесть смещение из-за первого столбца.
            res_col = int((positive_columns[0] + 1).item())

            # Ищем минимальное отношение `Si0 / x[res_col]` (Идём по всем, кроме ЦФ ищём минимальное отношение).
            s_i0 = main_table[:-1, 0]
            curr = main_table[:-1, res_col]
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
            res_element = main_table[res_row, res_col]
            _logger.info("Разрешающая строка: %s", self.left_column_[res_row])
            _logger.info("Разрешающий столбец: %s", self.top_row_[res_col])

            # Пересчёт симплекс-таблицы.
            recalculated_table = main_table - (
                    main_table[:, res_col][:, np.newaxis] * main_table[res_row, :]
            ) / res_element

            # Пересчёт разрешающей строки.
            for j in range(main_table.shape[1]):
                if j != res_col:
                    recalculated_table[res_row][j] = main_table[res_row][j] / res_element

            # Пересчёт разрешающего столбца.
            for i in range(main_table.shape[0]):
                if i != res_row:
                    recalculated_table[i][res_col] = -(main_table[i][res_col] / res_element)

            # Пересчёт разрешающего элемента.
            recalculated_table[res_row][res_col] = 1 / res_element

            main_table = recalculated_table
            self.swap_headers(res_row, res_col)
            _logger.info("%s", self)

        # Если задача на max, то в начале свели задачу к поиску min, а теперь
        # возьмём это решение со знаком минус и получим ответ для max.
        if extremum == "max":
            main_table[main_table.shape[0] - 1][0] *= -1

        _logger.info("Оптимальное решение найдено!")
        self.main_table_ = cp.asnumpy(main_table)
