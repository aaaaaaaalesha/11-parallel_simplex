"""Пользовательские исключения для симплекс-метода."""


class SimplexProblemException(Exception):
    """Для решения прямой задачи симплекс-методом."""
    pass


class DualProblemException(SimplexProblemException):
    """Для решения двойственной задачи симплекс-методом."""
    pass


class AlreadySolvedException(SimplexProblemException):
    """Исключение для ограничения попытки повторного запуска алгоритмов."""
    pass
