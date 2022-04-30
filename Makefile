# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

MODEL_PREFIX=model
AT_INPUT_WIDTH=224
AT_INPUT_HEIGHT=224
AT_INPUT_COLORS=3

LOAD_QUANTIZATION= #-q #to load a tflite quantized model
IMAGE=$(CURDIR)/samples/1_0.pgm

io=host


RAM_FLASH_TYPE ?= HYPER
#PMSIS_OS=freertos
APP_CFLAGS += -DUSE_HYPER
MODEL_L3_EXEC=hram
MODEL_L3_CONST=hflash
#use tflite_q model
MODEL_QUANTIZED=

QUANT_BITS?=8
BUILD_DIR=BUILD
MODEL_SQ8=1
NNTOOL_SCRIPT=model/nntool_script
MODEL_SUFFIX = _SQ8BIT

$(info Building GAP8 mode with $(QUANT_BITS) bit quantization)

include model_decl.mk

# Here we set the memory allocation for the generated kernels
# REMEMBER THAT THE L1 MEMORY ALLOCATION MUST INCLUDE SPACE
# FOR ALLOCATED STACKS!
CLUSTER_STACK_SIZE=8192
CLUSTER_SLAVE_STACK_SIZE=1024
TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY=150000
MODEL_L3_MEMORY=8388608

pulpChip = GAP
PULP_APP = application

APP = application
APP_SRCS += application.c $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE)

READFS_FILES=$(abspath $(MODEL_TENSORS))

# all depends on the model
all:: model

clean:: clean_model

include model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

