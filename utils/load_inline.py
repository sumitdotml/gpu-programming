from torch.utils.cpp_extension import load_inline

my_module = load_inline(
    name="my_module",
    cpp_sources=["kernels/add/add.cpp"],
    functions="add",
    verbose=True,
)

print(my_module.add(1, 2))
