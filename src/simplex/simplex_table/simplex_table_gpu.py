import logging

import cupy as cp
import numpy as np
from cupy.cuda import Device

_logger = logging.getLogger(__name__)


class SimplexTableGPU:
    """Класс симплекс-таблицы с параллельной обработкой на GPU."""

    def __init__(self, obj_func_coffs: np.array, constraint_system_lhs: np.array, constraint_system_rhs: np.array):
        """
        :param obj_func_coffs: Коэффициенты ЦФ.
        :param constraint_system_lhs: Левая часть системы ограничений.
        :param constraint_system_rhs: Правая часть системы ограничений.
        """
        self.obj_func_coffs_ = cp.array(obj_func_coffs)  # загружаем на GPU
        self.constraint_system_lhs_ = cp.array(constraint_system_lhs)  # загружаем на GPU
        self.constraint_system_rhs_ = cp.array(constraint_system_rhs)  # загружаем на GPU

        var_count = len(obj_func_coffs)
        constraint_count = constraint_system_lhs.shape[0]

        # Формируем симплекс-таблицу.
        self.main_table_ = cp.zeros((constraint_count + 1, var_count + 1), dtype=cp.float64)
        self.main_table_[:constraint_count, 0] = self.constraint_system_rhs_  # Столбец Si0
        self.main_table_[-1, 1:] = -self.obj_func_coffs_  # Строка F

        for i in range(constraint_count):
            self.main_table_[i, 1:] = self.constraint_system_lhs_[i]

        # Метаданные для параллельных вычислений.
        self.num_gpus = cp.cuda.runtime.getDeviceCount()  # Получаем количество доступных GPU

        _logger.info("Используется %d GPU для параллелизации", self.num_gpus)

    def recalc_table_parallel(self, res_row: int, res_col: int, res_element: float):
        """
        Параллельный перерасчет симплекс-таблицы.
        :param res_row: Индекс разрешающей строки.
        :param res_col: Индекс разрешающего столбца.
        :param res_element: Разрешающий элемент.
        """
        # Перекладываем данные на все GPU и параллелим по столбцам.
        table_shape = self.main_table_.shape
        cols_per_device = (table_shape[1] - 1) // self.num_gpus

        # Распределение столбцов между GPU.
        devices = [Device(i) for i in range(self.num_gpus)]
        table_blocks = []

        for i, device in enumerate(devices):
            with device:
                # Каждый GPU получит часть столбцов
                start_col = i * cols_per_device + 1
                end_col = (i + 1) * cols_per_device + 1 if i != self.num_gpus - 1 else table_shape[1]
                sub_matrix = self.main_table_[:, start_col:end_col].copy()
                table_blocks.append(sub_matrix)

        # Пересчет на каждом GPU в параллельных потоках
        for i, device in enumerate(devices):
            with device:
                start_col = i * cols_per_device + 1
                end_col = (i + 1) * cols_per_device + 1 if i != self.num_gpus - 1 else table_shape[1]

                # Параллельные вычисления для части столбцов
                recalc_part = self._recalc_on_device(res_row, res_col, res_element,
                                                     self.main_table_[:, start_col:end_col])

                # Копируем результат обратно
                self.main_table_[:, start_col:end_col] = recalc_part

        # Пересчет столбцов Si0 и значения целевой функции
        self._recalc_left_column(res_row, res_col, res_element)

    @staticmethod
    def _recalc_on_device(res_row: int, res_col: int, res_element: float, sub_matrix):
        """
        Перерасчет симплекс-таблицы для отдельного блока столбцов на GPU.
        :param res_row: Индекс разрешающей строки.
        :param res_col: Индекс разрешающего столбца.
        :param res_element: Разрешающий элемент.
        :param sub_matrix: Подматрица, которая перерабатывается на GPU.
        """
        recalced_part = cp.zeros_like(sub_matrix)

        # Пересчет разрешающей строки
        recalced_part[res_row, :] = sub_matrix[res_row, :] / res_element

        # Пересчет остальных элементов
        for i in range(sub_matrix.shape[0]):
            if i != res_row:
                recalced_part[i, :] = sub_matrix[i, :] - sub_matrix[res_row, :] * (sub_matrix[i, res_col] / res_element)

        return recalced_part

    def _recalc_left_column(self, res_row: int, res_col: int, res_element: float):
        """
        Перерасчет левого столбца таблицы и строки целевой функции.
        """
        # Столбец Si0
        self.main_table_[:, 0] = cp.where(
            cp.arange(self.main_table_.shape[0]) != res_row,
            self.main_table_[:, 0] - (self.main_table_[:, res_col] * self.main_table_[res_row, 0] / res_element),
            self.main_table_[:, 0] / res_element
        )

        # Обновление строки целевой функции.
        self.main_table_[-1, 0] = -self.main_table_[:, 0].sum()
