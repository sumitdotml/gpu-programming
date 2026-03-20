from torch.utils.cpp_extension import load_inline

add_cpp = """
#include <iostream>
int add(int a, int b) {
    return a + b;
}
"""


my_module = load_inline(
    name="my_module",
    cpp_sources=[add_cpp],
    functions="add",
    verbose=True,
)

print(my_module.add(1, 2))
