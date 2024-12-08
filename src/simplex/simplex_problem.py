"""Задача ЛП и решение симплекс-методом."""
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .exceptions import SimplexProblemException
from .simplex_table import BaseSimplexTable, CupySimplexTable
from .types import Solution, TargetFunctionValue, ValueType, VariableNames, VariableValues

_logger = logging.getLogger(__name__)


class SimplexProblem:
    """
    Класс для решения задачи ЛП симплекс-методом.
    """

    def __init__(self, input_path: Path, use_gpu=False, verbose=True):
        """
        Регистрирует входные данные из JSON-файла. Определяет условие задачи.
        :param input_path: Путь до JSON-файла с входными данными.
        :param use_gpu: Флаг, позволяющий выключить параллельное вычисление на GPU.
        :param verbose: Флаг
        """
        # Парсим JSON-файл с входными данными
        with input_path.open() as read_file:
            json_data = json.load(read_file)

        # Вектор-строка с - коэффициенты ЦФ.
        self.obj_func_coffs_ = np.array(json_data["obj_func_coffs"])
        # Матрица ограничений А.
        self.constraint_system_lhs_ = np.array(json_data["constraint_system_lhs"])
        # Вектор-столбец ограничений b.
        self.constraint_system_rhs_ = np.array(json_data["constraint_system_rhs"])
        # Направление задачи (min или max)
        self.func_direction_ = json_data["func_direction"]
        # Найденное решение задачи (вектор значений переменных и значение целевой функции).
        self.solution: Solution | None = None
        # Выводить ли результаты найденного решения в консоль.
        self._verbose = verbose

        if len(self.constraint_system_rhs_) != self.constraint_system_rhs_.shape[0]:
            raise SimplexProblemException("Ошибка при вводе данных. "
                                          "Число строк в матрице и столбце ограничений не совпадает.")

        # Если задача на max, то меняем знаки ЦФ и направление задачи
        # (в конце возьмем решение со знаком минус и получим искомое).
        if self.func_direction_ == "max":
            self.obj_func_coffs_ *= -1

        _logger.info("%s", self)

        # Выбор класса в зависимости от того, хотим ли мы использовать GPU.
        simplex_table_backend = CupySimplexTable if use_gpu else BaseSimplexTable
        _logger.info(f"Используем %s (GPU: %s)", simplex_table_backend, use_gpu)

        # Инициализация симплекс-таблицы.
        self.simplex_table_ = simplex_table_backend(
            obj_func_coffs=self.obj_func_coffs_,
            constraint_system_lhs=self.constraint_system_lhs_,
            constraint_system_rhs=self.constraint_system_rhs_,
        )

    @classmethod
    def from_constraints(
            cls,
            obj_func_coffs: list,
            constraint_system_lhs: list,
            constraint_system_rhs: list,
            func_direction="max",
            use_gpu=False,
            verbose=True,
    ) -> "SimplexProblem":
        """
        Альтернативный конструктор, использующий входные значения напрямую.
        :param obj_func_coffs: Вектор-строка с - коэффициенты ЦФ.
        :param constraint_system_lhs: Матрица ограничений А.
        :param constraint_system_rhs: Вектор-столбец ограничений b.
        :param func_direction: Направление задачи ("min" (default) или "max").
        :param use_gpu: Флаг, позволяющий выключить параллельное вычисление на GPU.
        :param verbose:
        :return: Экземпляр класса SimplexProblem.
        """
        with NamedTemporaryFile(mode="w") as input_file:
            input_path = Path(input_file.name)
            input_path.write_text(
                json.dumps(
                    {
                        "obj_func_coffs": obj_func_coffs,
                        "constraint_system_lhs": constraint_system_lhs,
                        "constraint_system_rhs": constraint_system_rhs,
                        "func_direction": func_direction,
                    }
                )
            )
            return cls(input_path, use_gpu, verbose)

    def __str__(self):
        """Вывод условия прямой задачи ЛП."""
        return "\n".join(
            (
                f"F = c⋅x -> {self.func_direction_},",
                "Ax <= b,\nx1,x2, ..., xn >= 0",
                f"c^T = {self.obj_func_coffs_},",
                f"A =\n{self.constraint_system_lhs_},",
                f"b^T = {self.constraint_system_rhs_}.",
            )
        )

    def __repr__(self):
        """Условие задачи для отображения в Jupyter."""
        return str(self)

    @property
    def dummy_variables(self) -> tuple[VariableNames, VariableValues]:
        """Имена и значения фиктивных переменных."""
        dummy_variables_names: list[str] = self.simplex_table_.top_row_[2:]
        return dummy_variables_names, [0] * len(dummy_variables_names)  # noqa

    def solve(self) -> Solution:
        """
        Запуск решения задачи.
        :returns: Оптимальное решение задачи ЛП: вектор значений переменных и значение целевой функции.
        """
        if self.solution:
            var_values, target_value = self.solution
            var_values_literal: str = ", ".join(f"{var_value:.3f}" for var_value in var_values)
            _logger.warning(f"Решение задачи уже получено: ({var_values_literal}); F = {target_value:.3f}")
            return self.solution

        _logger.info("Процесс решения:")
        self._reference_solution()
        return self._optimal_solution()

    def _reference_solution(self):
        """Этап 1: Поиск опорного решения."""
        _logger.info("Поиск опорного решения: \nИсходная симплекс-таблица:\n%s", self.simplex_table_)
        while not self.simplex_table_.is_find_ref_solution():
            self.simplex_table_.search_ref_solution()

        _logger.info("Опорное решение найдено!")
        return self.__output_solution() if self._verbose else None

    def _optimal_solution(self) -> Solution:
        """Этап 2: Поиск оптимального решения."""
        _logger.info("Поиск оптимального решения:")
        while not self.simplex_table_.is_find_opt_solution():
            self.simplex_table_.optimize_ref_solution()

        # Если задача на max, то в начале свели задачу к поиску min, а теперь
        # возьмём это решение со знаком минус и получим ответ для max.
        if self.func_direction_ == "max":
            table_rows_count: int = self.simplex_table_.main_table_.shape[0]
            self.simplex_table_.main_table_[table_rows_count - 1][0] *= -1

        self.solution: Solution = self.__collect_solution()
        _logger.info("Оптимальное решение найдено!")
        if self._verbose:
            self.__output_solution()
        return self.solution

    def __output_solution(self) -> None:
        """
        Метод выводит текущее решение. Используется для вывода опорного и оптимального решений.
        """
        dummy_vars: tuple[VariableNames, VariableValues] = self.dummy_variables
        # Выводим фиктивные переменные со значениями, равными нулю.
        _logger.info("".join([*[f"{var} = " for var in dummy_vars[0]], "0, "]))

        last_row_ind: int = self.simplex_table_.main_table_.shape[0] - 1
        # Выводим оставшиеся переменные и их значения.
        _logger.info(", ".join(
            [
                f"{self.simplex_table_.left_column_[i]} = {self.simplex_table_.main_table_[i][0]:.3f}"
                for i in range(last_row_ind)
            ]
        ))
        # Выводим значение целевой функции.
        _logger.info(f"Целевая функция: F = {self.simplex_table_.main_table_[last_row_ind][0]:.3f}")

    def __collect_solution(self) -> Solution:
        """
        Формирует решение задачи ЛП симплекс-методом.
        :returns Solution: Найденные оптимальные значения переменных и целевой функции.
        """
        # Заполняем словарь с именами переменных и их значениями.
        vars_to_values: dict[str, ValueType] = {
            dummy_var_name: 0 for dummy_var_name in self.simplex_table_.top_row_[2:]
        }
        last_row_ind: int = self.simplex_table_.main_table_.shape[0] - 1
        for i in range(last_row_ind):
            vars_to_values[self.simplex_table_.left_column_[i]] = self.simplex_table_.main_table_[i][0]

        # Расставляем значения по местам в векторе значений [x1, x2, ..., xn].
        values_vector: VariableValues = [0] * len(vars_to_values)
        for var_name, var_value in vars_to_values.items():
            values_vector[int(var_name[1:]) - 1] = var_value

        # Возвращаем сформированное решение.
        target_function_value: TargetFunctionValue = self.simplex_table_.main_table_[last_row_ind][0]
        return values_vector, target_function_value
