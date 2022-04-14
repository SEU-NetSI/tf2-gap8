#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"





void mnistModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 2, "CNN_BasicKernels_SQ8.h", "mnist.h");
    SetGeneratedFilesNames("mnistKernels.c", "mnistKernels.h");


    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "mnist_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "mnist_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "mnist_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "mnist_L3_Flash", "mnist_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();


    // generator for input_1_formatter
    CNN_Norm("S1_Op_input_1_formatter", 28, 28, 1, KOP_NORM_BW);
    // generator for CONV_2D_0_0_fusion
    CNN_ConvolutionPoolAct_SQ8("S4_Conv2d_32x1x5x5_MaxPool_2x2_Relu", 0, 4, 1,
                               1, 32, 28, 28,
                               KOP_CONV, 5, 5, 1, 1, 1, 1, 0,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for CONV_2D_0_2_fusion
    CNN_ConvolutionPoolAct_SQ8("S7_Conv2d_64x32x3x3_MaxPool_2x2_Relu", 0, 4, 1,
                               32, 64, 12, 12,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 0,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);
    
    // generator for FULLY_CONNECTED_0_5_fusion
    CNN_LinearAct_SQ8("S10_Op_FULLY_CONNECTED_0_5_fusion", 0,
                      4, 1,
                      1600, 64,
                      KOP_LINEAR, KOP_RELU);
    
    // generator for FULLY_CONNECTED_0_6_fusion
    CNN_LinearAct_SQ8("S13_Op_FULLY_CONNECTED_0_6_fusion", 0,
                      4, 1,
                      64, 10,
                      KOP_LINEAR, KOP_RELU);
    
    // generator for SOFTMAX_0_7
    CNN_SoftMax_SQ8("S14_SoftMax", 0, 10, KOP_SOFTMAX);

#define GRAPH
#ifdef GRAPH
    CreateGraph("mnistCNN",
        /* Arguments either passed or globals */
            CArgs(23,
                TCArgInfo("unsigned char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Sequentialconv2dconv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialconv2dconv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2dbias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Conv2dbias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S4_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S4_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S4_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialconv2d_1conv2d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialconv2d_1conv2d.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Conv2d_1bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Conv2d_1bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S7_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S7_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S7_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialdensematmul", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialdensematmul.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Densebias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Densebias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Sequentialdense_1matmul", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Sequentialdense_1matmul.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Dense_1bias", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/Dense_1bias.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // BiasQ: 0all 0
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S13_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.250000 out: 0.000031 NORM: 13
                TCArgInfo("signed char * __restrict__", "S14_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("BUILD_MODEL_SQ8BIT/tensors/S14_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed short * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(5,
            TCArgInfo("signed char * __restrict__", "S1_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S10_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S13_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node input_1_formatter inq 0.00<(u8-0.00)*1.00000000<255.00 outq -1.00<(i8-0.00)*0.00787402<1.00
    AddNode("S1_Op_input_1_formatter",
        Bindings(2,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_OUT, "S1_Output", 0)
        )
    );
    // Node S4_Conv2d_32x1x5x5_MaxPool_2x2_Relu inq -1.00<(i8-0.00)*0.00787402<1.00 weightsq chan<(i8-0.00)*chan<chan outq -4.63<(i8-0.00)*0.03617034<4.59 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S4_Conv2d_32x1x5x5_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S1_Output", 0),
            GNodeArg(GNA_IN, "Sequentialconv2dconv2d", 0),
            GNodeArg(GNA_IN, "Conv2dbias", 0),
            GNodeArg(GNA_OUT, "S4_Output", 0),
            GNodeArg(GNA_IN, "S4_Mul_scale", 0),
            GNodeArg(GNA_IN, "S4_Mul_shift", 0),
            GNodeArg(GNA_IN, "S4_Infos", 0)
        )
    );
    // Node S7_Conv2d_64x32x3x3_MaxPool_2x2_Relu inq -4.63<(i8-0.00)*0.03617034<4.59 weightsq chan<(i8-0.00)*chan<chan outq -8.58<(i8-0.00)*0.06701219<8.51 biasesq chan<(i32-0.00)*chan<chan
    AddNode("S7_Conv2d_64x32x3x3_MaxPool_2x2_Relu",
        Bindings(7,
            GNodeArg(GNA_IN, "S4_Output", 0),
            GNodeArg(GNA_IN, "Sequentialconv2d_1conv2d", 0),
            GNodeArg(GNA_IN, "Conv2d_1bias", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Mul_scale", 0),
            GNodeArg(GNA_IN, "S7_Mul_shift", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_5 inq -8.58<(i8-0.00)*0.06701219<8.51 weightsq chan<(i8-0.00)*chan<chan outq -12.29<(i8-0.00)*0.09604139<12.20
    AddNode("S10_Op_FULLY_CONNECTED_0_5_fusion",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "Sequentialdensematmul", 0),
            GNodeArg(GNA_IN, "Densebias", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_6 inq -12.29<(i8-0.00)*0.09604139<12.20 weightsq chan<(i8-0.00)*chan<chan outq -32.00<(i8-0.00)*0.25000000<31.75 forced
    AddNode("S13_Op_FULLY_CONNECTED_0_6_fusion",
        Bindings(7,
            GNodeArg(GNA_IN, "S10_Output", 0),
            GNodeArg(GNA_IN, "Sequentialdense_1matmul", 0),
            GNodeArg(GNA_IN, "Dense_1bias", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node SOFTMAX_0_7 inq 2 outq 15
    AddNode("S14_SoftMax",
        Bindings(3,
            GNodeArg(GNA_IN, "S13_Output", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S14_Infos", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    mnistModel(64000, 300000, 8000000, 20*1024*1024);
    GenerateTilingCode();
    return 0;
}
