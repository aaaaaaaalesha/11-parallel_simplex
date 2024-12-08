import numpy as np
from numba import cuda


@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = x[idx] + y[idx]


n = 1000000
x = np.ones(n)
y = np.ones(n)
out = np.zeros(n)

# Переносим данные на устройство
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.to_device(out)

# Запускаем ядро
threads_per_block = 256
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# Копируем результат обратно на хост
d_out.copy_to_host(out)

print(out)  # Должно вывести массив из единиц
