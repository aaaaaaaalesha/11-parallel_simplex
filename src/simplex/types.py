"""Подсказки типов для симплекс-метода."""

from typing import Sequence, TypeAlias

# Type to annotate values in simplex problems.
ValueType: TypeAlias = int | float

# Type to annotate variable values vector.
VariableNames: TypeAlias = Sequence[str]

# Type to annotate variable values vector.
VariableValues: TypeAlias = list[ValueType, ...]

# Type to annotate target function value.
TargetFunctionValue: TypeAlias = ValueType

# Type to annotate solution of simplex problem.
Solution = tuple[VariableValues, TargetFunctionValue]
