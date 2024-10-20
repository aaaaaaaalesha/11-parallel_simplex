"""Двойственная задача ЛП и решение симплекс-методом."""
import json
import logging
from pathlib import Path

import numpy as np

from .exceptions import DualProblemException
from .simplex_problem import SimplexProblem
from .simplex_table import SimplexTable
from .types import Solution

_logger = logging.getLogger(__name__)


class DualProblem(SimplexProblem):
    """
    Класс унаследован от Simplex для переформулирования задачи из ПЗ в ДЗ.
    """

    def __init__(self, input_path: Path):
        """
        Регистрирует входные данные из JSON-файла. Определяет условие двойственной задачи.
        :param Path input_path: Путь до JSON-файла с входными данными.
        """

        # Парсим JSON-файл с входными данными
        with input_path.open() as read_file:
            input_data: dict = json.load(read_file)

        # Коэффициенты при ЦФ в ДЗ равны свободным членам ограничений в ПЗ.
        self.obj_func_coffs_ = np.array(input_data["constraint_system_rhs"])
        # Свободные члены ограничений в ДЗ равны коэффициентам при ЦФ в ПЗ.
        self.constraint_system_lhs_ = np.array(input_data["constraint_system_lhs"]).transpose()
        # Коэффициенты любого ограничения ДЗ равны коэффициентам при одной переменной из всех ограничений ПЗ.
        self.constraint_system_rhs_ = np.array(input_data["obj_func_coffs"])
        # Минимизация ЦФ в ПЗ соответствует максимизации ЦФ в ДЗ.
        self.func_direction_ = "max" if input_data["func_direction"] == "min" else "min"
        # Найденное решение задачи (вектор значений переменных и значение целевой функции).
        self.solution: Solution | None = None

        _logger.info(str(self))

        # Ограничения вида (<=) ПЗ переходят в ограничения вида (>=) ДЗ.
        self.constraint_system_lhs_ *= -1
        self.constraint_system_rhs_ *= -1

        if len(self.constraint_system_rhs_) != self.constraint_system_rhs_.shape[0]:
            exc_msg = "Ошибка при вводе данных. Число строк в матрице и столбце ограничений не совпадает"
            raise DualProblemException(exc_msg)

        if len(self.constraint_system_rhs_) > len(self.obj_func_coffs_):
            exc_msg = "СЛАУ несовместна! Число уравнений больше числа переменных"
            raise DualProblemException(exc_msg)

        # Если задача на max, то
        # меняем знаки ЦФ и направление задачи
        # (в конце возьмем решение со знаком минус и получим искомое).
        if self.func_direction_ == "max":
            self.obj_func_coffs_ *= -1

        # Инициализация симплекс-таблицы.
        self.simplex_table_ = SimplexTable(
            self.obj_func_coffs_,
            self.constraint_system_lhs_,
            self.constraint_system_rhs_,
        )

    def __str__(self):
        """Вывод условия двойственной задачи ЛП."""
        multiplier: int = -1 if self.func_direction_ == "max" else 1
        return "\n".join(
            [
                f"F = b^T⋅x -> {self.func_direction_},",
                "A^T⋅x >= c^T,",
                "x1, x2, ..., xn >= 0",
                f"b^T = {multiplier * self.obj_func_coffs_}",
                f"A^T =\n{self.constraint_system_lhs_},",
                f"c^T = {self.constraint_system_rhs_}.",
            ]
        )