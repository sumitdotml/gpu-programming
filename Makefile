NVCC ?= nvcc
CUDAFLAGS ?= -O2
SRC ?= pmpp/add.cu
BUILD_DIR := build
OUT := $(BUILD_DIR)/$(basename $(notdir $(SRC))).out
GPU ?= RTX-PRO-6000
MODAL_IMAGE ?= nvidia/cuda:12.8.1-devel-ubuntu24.04

.PHONY: run modal clean

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: | $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) $(SRC) -o $(OUT)
	./$(OUT)

modal:
	modal shell --gpu $(GPU) --image $(MODAL_IMAGE) --add-local $(SRC) -c 'nvcc $(CUDAFLAGS) /mnt/$(notdir $(SRC)) -o /tmp/$(notdir $(OUT)) && /tmp/$(notdir $(OUT))'

clean:
	rm -rf $(BUILD_DIR)
