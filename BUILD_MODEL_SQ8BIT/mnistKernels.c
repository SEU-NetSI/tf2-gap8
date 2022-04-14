#include "mnistKernels.h"
L1_CL_MEM AT_L1_POINTER mnist_L1_Memory;
L2_MEM AT_L2_POINTER mnist_L2_Memory;
static AT_HYPERFLASH_FS_T HyperFlash;
void S1_Op_input_1_formatter(
		unsigned char * __restrict__ In,
		signed char * __restrict__ Out)

{
	/* Shared L1: 1568 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaW_Evt1;
	KerNormBW_fps_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 784 [Tile0, 1:[28x28], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[28x28], 1]
		Tile0: [0, 784, 784], Tile1: [0, 784, 784], Tile2; [0, 784, 784]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 784 [Tile0, 1:[28x28], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[28x28], 1]
		Tile0: [0, 784, 784], Tile1: [0, 784, 784], Tile2; [0, 784, 784]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (unsigned char *__restrict__) (mnist_L1_Memory+0);
	KerArg0->Out = (signed char *__restrict__) (mnist_L1_Memory+784);
	KerArg0->W = (unsigned short int) (28);
	KerArg0->H = (unsigned short int) (28);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0), 784, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		AT_FORK(gap_ncore(), (void *) CNN_NormBW_offset_fps, (void *) KerArg0);
		__CALL(CNN_NormBW_offset_fps, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+784), 784, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S4_Conv2d_32x1x5x5_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 43372 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last;
	int T0Ind, T0Ind_Total=0, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _C_Out;
	unsigned int _SP_Out, _SC_Out;
	unsigned int _LP_Out, _LC_Out;
	unsigned int _N_In;
	unsigned int _SN_In;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 32, Tiled: 1][Tile0 Dim: 2][D0 Dim: Init: 1, Tiled: 1]
	Ker Arg: Out, Tiled Space: Tile0
		Min Pipe Depth: -1, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 4608 [D1, [0 x 4608, 4608]][Tile0, 2:[12x6, 12x6], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 4608, 4608]][Tile0, 2:[12x6, 12x6], 1]
		Tile0: [0, 2304, 72], Tile1: [72, 2304, 72], Tile2; [0, 2304, 72]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 128 [D1, [0 x 128, 128]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 128, 128]]
		Tile0: [0, 128, 128], Tile1: [0, 128, 128], Tile2; [0, 128, 128]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 32 [D1, [0 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 32, 32]]
		Tile0: [0, 32, 32], Tile1: [0, 32, 32], Tile2; [0, 32, 32]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 800 [D1, [0 x 800, 800]][D0, [0 x 800, 800]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 800, 800]][D0, [0 x 800, 800]]
		Tile0: [0, 800, 800], Tile1: [0, 800, 800], Tile2; [0, 800, 800]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 2 physical tiles
			Total Size: 784 [D0, [0 x 784, 784]][Tile0, 2:[28x16, 28x16], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[28x16], 1][D0, [0 x 784, 784]]
		Tile0: [0, 448, 448], Tile1: [336, 448, 448], Tile2; [0, 448, 448]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 73728 [D1, [0 x 73728, 73728]][Tile0, 2:[24x12, 24x12], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 73728, 73728]][Tile0, 2:[24x12, 24x12], 4]
		Tile0: [0, 36864, 1152], Tile1: [0, 36864, 1152], Tile2; [0, 36864, 1152]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 2 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 2:[9x1, 9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 2:[9x1, 9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (mnist_L1_Memory+6496);
	KerArg0->W = (unsigned short int) (24);
	KerArg0->H = (unsigned short int) (12);
	KerArg0->Feat = (unsigned short int) (32);
	KerArg0->Bias = (void * __restrict__) (mnist_L1_Memory+896);
	KerArg1->W = (unsigned short int) (28);
	KerArg1->UsedW = (unsigned short int) (28);
	KerArg1->H = (unsigned short int) (16);
	KerArg1->UsedH = (unsigned short int) (16);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->TotalInFeatures = (unsigned short int) (1);
	KerArg1->Filter = (signed char * __restrict__) (mnist_L1_Memory+1088);
	KerArg1->Out = (int * __restrict__) (mnist_L1_Memory+6496);
	KerArg1->Pad = (v4s) 0;
	KerArg2->In = (int *__restrict__) (mnist_L1_Memory+6496);
	KerArg2->Out = (void *__restrict__) (mnist_L1_Memory+6496);
	KerArg2->Feat = (unsigned short int) (32);
	KerArg2->W = (unsigned short int) (24);
	KerArg2->H = (unsigned short int) (12);
	KerArg2->Scale = (unsigned char *__restrict__) (mnist_L1_Memory+1024);
	KerArg2->ScaleN = (unsigned char *__restrict__) (mnist_L1_Memory+1056);
	KerArg2->Infos = (signed char *__restrict__) (mnist_L1_Memory+43360);
	KerArg3->In = (signed char * __restrict__) (mnist_L1_Memory+6496);
	KerArg3->W = (unsigned short int) (24);
	KerArg3->UsedW = (unsigned short int) (24);
	KerArg3->H = (unsigned short int) (12);
	KerArg3->UsedH = (unsigned short int) (12);
	KerArg3->Feat = (unsigned short int) (32);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (mnist_L1_Memory+43360);
	/*================================= Read Tiles Prolog ===============================*/
	_C_Out=0; _SC_Out=2304; _LC_Out=72;
	_SP_Out=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+896), 128, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1024), 32, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1056), 32, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1088), 800, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0+0), 448, 0, &DmaR_Evt5);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+43360), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1;
		for (T0Ind=0; T0Ind<2; T0Ind++, T0Ind_Total++) { /* Iteration on Tile0 */
			int T0Ind_Last = (T0Ind==1), T0Ind_NextLast = ((T0Ind+1)==1);
			/*================================= Prepare Tiles ===================================*/
			_SN_In = 0;
			if (!(T0Ind_Last)) {
				_N_In = _N_In + (336); _SN_In = (1*(448)); 
			} else if (!(1)) {
				_N_In = _N_In + (-336); _SN_In = (1*(448)); 
			}
			/*============================= End Prepare Tiles ===================================*/
			/*================================= Read Tiles ======================================*/
			AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read In */
			if (_SN_In) {
				AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0+448*((T0Ind_Total+1)%2)),
						_SN_In, 0, &DmaR_Evt5);
			}
			/*============================= End Read Tiles ======================================*/
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(mnist_L1_Memory+43360))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			{ /* Single iteration on D0 */
				int D0Ind_Last = 1;
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (mnist_L1_Memory+0+448*((T0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv5x5Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv5x5Stride1_SQ8, KerArg1);
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			KerArg3->Out = (signed char * __restrict__) (mnist_L1_Memory+1888+2304*((T0Ind_Total)%2));
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
			/*================================= Write Tiles =====================================*/
			if (_SP_Out) AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
			AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) Out+_C_Out), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1888+2304*((T0Ind_Total)%2)),
					_SC_Out, 144, _LC_Out, 1, &DmaW_Evt1);
			/*============================= End Write Tiles =====================================*/
			/*================================= Update Arg Pipeline =============================*/
			_SP_Out = _SC_Out;_LP_Out = _LC_Out;
			/*============================= End Update Arg Pipeline =============================*/
			/*================================= Prepare Tiles ===================================*/
			_SC_Out = 0;
			if (!(T0Ind_Last)) {
				_C_Out = _C_Out + (72); _LC_Out = (72); _SC_Out = (32*_LC_Out); 
			}
			/*============================= End Prepare Tiles ===================================*/
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait previous DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S7_Conv2d_64x32x3x3_MaxPool_2x2_Relu(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 44876 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerSetBias_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerConv_SQ8_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerConvLinReduct_SQ8_T S_KerArg2, *KerArg2 = &S_KerArg2;
	KerPool_SQ8_T S_KerArg3, *KerArg3 = &S_KerArg3;

	/* Iteration space related variables */
	int D1Ind, D1Ind_Last, D1Ind_NextLast;
	int T0Ind, T0Ind_Last, T0Ind_NextLast;
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	/* User kernel arguments related variables */
	unsigned int _N_In;
	unsigned int _SN_In;
	unsigned int _LN_In;
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D1 Dim: Init: 64, Tiled: 1][Tile0 Dim: 1][D0 Dim: Init: 32, Tiled: 3]
	Ker Arg: In, Tiled Space: Tile0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 4608 [D0, [2 x 1728, 1152]][Tile0, 1:[12x12], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[12x12, 1:12x12, 12x12], 1][D0, [2 x 1728, 1152]]
		Tile0: [0, 1728, 144], Tile1: [1728, 1728, 144], Tile2; [3456, 1152, 144]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 256 [D1, [0 x 256, 256]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 256, 256]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [D1, [0 x 64, 64]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 64, 64]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 3 logical tiles, 3 physical tiles
			Total Size: 18432 [D1, [0 x 18432, 18432]][D0, [2 x 6912, 4608]]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 18432, 18432]][D0, [2 x 6912, 4608]]
		Tile0: [0, 6912, 6912], Tile1: [6912, 6912, 6912], Tile2; [13824, 4608, 4608]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1600 [D1, [0 x 1600, 1600]][Tile0, 1:[5x5], 1]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 1600, 1600]][Tile0, 1:[5x5], 1]
		Tile0: [0, 1600, 1600], Tile1: [0, 1600, 1600], Tile2; [0, 1600, 1600]
	Ker Arg: ConvOut, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 25600 [D1, [0 x 25600, 25600]][Tile0, 1:[10x10], 4]
		KerArgItSpace (User Kernel Iter Order):
			[D1, [0 x 25600, 25600]][Tile0, 1:[10x10], 4]
		Tile0: [0, 25600, 400], Tile1: [0, 25600, 400], Tile2; [0, 25600, 400]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->Out = (int * __restrict__) (mnist_L1_Memory+19264);
	KerArg0->W = (unsigned short int) (10);
	KerArg0->H = (unsigned short int) (10);
	KerArg0->Feat = (unsigned short int) (64);
	KerArg0->Bias = (void * __restrict__) (mnist_L1_Memory+3456);
	KerArg1->W = (unsigned short int) (12);
	KerArg1->UsedW = (unsigned short int) (12);
	KerArg1->H = (unsigned short int) (12);
	KerArg1->UsedH = (unsigned short int) (12);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Out = (int * __restrict__) (mnist_L1_Memory+19264);
	KerArg1->Pad = (v4s) 0;
	KerArg2->In = (int *__restrict__) (mnist_L1_Memory+19264);
	KerArg2->Out = (void *__restrict__) (mnist_L1_Memory+19264);
	KerArg2->Feat = (unsigned short int) (64);
	KerArg2->W = (unsigned short int) (10);
	KerArg2->H = (unsigned short int) (10);
	KerArg2->Scale = (unsigned char *__restrict__) (mnist_L1_Memory+3712);
	KerArg2->ScaleN = (unsigned char *__restrict__) (mnist_L1_Memory+3776);
	KerArg2->Infos = (signed char *__restrict__) (mnist_L1_Memory+44864);
	KerArg3->In = (signed char * __restrict__) (mnist_L1_Memory+19264);
	KerArg3->W = (unsigned short int) (10);
	KerArg3->UsedW = (unsigned short int) (10);
	KerArg3->H = (unsigned short int) (10);
	KerArg3->UsedH = (unsigned short int) (10);
	KerArg3->Feat = (unsigned short int) (64);
	KerArg3->Out = (signed char * __restrict__) (mnist_L1_Memory+17664);
	KerArg3->Pad = (v4s) 0;
	KerArg3->PoolMax = (unsigned char) (1);
	KerArg3->DoScale = (unsigned char) (0);
	KerArg3->Infos = (signed char * __restrict__) (mnist_L1_Memory+44864);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0+0), 1728, 144, 144, 0, &DmaR_Evt1);
	_N_In=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+3456), 256, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+3712), 64, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+3776), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+3840+0), 6912, 0, &DmaR_Evt5);
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+44864), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D1 */
		int D1Ind_Last = 1, D1Ind_NextLast = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1, T0Ind_NextLast = 1;
			/*====================== Call Kernel LOC_D0_PROLOG =========================*/
			KerArg0->NormBias = (unsigned char) (((char *)(mnist_L1_Memory+44864))[5]);
			AT_FORK(gap_ncore(), (void *) KerParSetBiasB32_SQ8, (void *) KerArg0);
			__CALL(KerParSetBiasB32_SQ8, KerArg0);
			for (D0Ind=0; D0Ind<3; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
				int D0Ind_Last = (D0Ind==2), D0Ind_NextLast = ((D0Ind+1)==2);
				/*================================= Prepare Tiles ===================================*/
				_SN_In = 0;
				if (!(D0Ind_Last)) {
					_N_In = _N_In + (1728); _LN_In = (144); _SN_In = (((D0Ind_NextLast)?8:12)*_LN_In); 
				} else if (!(1)) {
					_N_In = _N_In + (-3456); _LN_In = (144); _SN_In = (12*_LN_In); 
				}
				_SN_Filter = 0;
				if (!(D0Ind_Last)) {
					_N_Filter = _N_Filter + ((6912)); _SN_Filter = (((1)?(((D0Ind_NextLast)?4608:6912)):(((D0Ind_NextLast)?4608:6912)))); 
				} else if (!((1))) {
					_N_Filter = _N_Filter + ((-13824)); _SN_Filter = (((1)?(6912):(6912))); 
				}
				/*============================= End Prepare Tiles ===================================*/
				/*================================= Read Tiles ======================================*/
				AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
				if (_SN_In) {
					AT_L2_COPY2D(0, ((AT_L2_EXT_ADDR_TYPE) In+_N_In), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0+1728*((D0Ind_Total+1)%2)),
							_SN_In, 144, _LN_In, 0, &DmaR_Evt1);
				}
				AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read Filter */
				if (_SN_Filter) {
					AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+3840+6912*((D0Ind_Total+1)%2)),
							_SN_Filter, 0, &DmaR_Evt5);
				}
				/*============================= End Read Tiles ======================================*/
				/*====================== Call Kernel LOC_D0 =========================*/
				KerArg1->In = (signed char * __restrict__) (mnist_L1_Memory+0+1728*((D0Ind_Total)%2));
				KerArg1->InFeatures = (unsigned short int) ((D0Ind_Last)?8:12);
				KerArg1->TotalInFeatures = (unsigned short int) ((D0Ind_Last)?8:12);
				KerArg1->Filter = (signed char * __restrict__) (mnist_L1_Memory+3840+6912*((D0Ind_Total)%2));
				AT_FORK(gap_ncore(), (void *) KerParConv3x3Stride1_SQ8, (void *) KerArg1);
				__CALL(KerParConv3x3Stride1_SQ8, KerArg1);
				/*================================= Update Arg Pipeline =============================*/
				/*============================= End Update Arg Pipeline =============================*/
			} /* End iteration on D0 */
			/*====================== Call Kernel LOC_D0_EPILOG =========================*/
			AT_FORK(gap_ncore(), (void *) KerParReductIO_CC_SQ8, (void *) KerArg2);
			__CALL(KerParReductIO_CC_SQ8, KerArg2);
			AT_FORK(gap_ncore(), (void *) KerParPool2x2Stride2_ReLU_SQ8, (void *) KerArg3);
			__CALL(KerParPool2x2Stride2_ReLU_SQ8, KerArg3);
		} /* End iteration on Tile0 */
	} /* End iteration on D1 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+17664), 1600, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S10_Op_FULLY_CONNECTED_0_5_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 27660 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerLinear_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Total=0, D0Ind_Last, D0Ind_NextLast;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	unsigned int _N_Filter;
	unsigned int _SN_Filter;
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 64, Tiled: 8][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 1600 [Tile0, 1:[1x1], 1600]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 1600]
		Tile0: [0, 1600, 1600], Tile1: [0, 1600, 1600], Tile2; [0, 1600, 1600]
	Ker Arg: Filter, Tiled Space: D0
		Min Pipe Depth: 0, Max Pipe Depth: 1
		KerArgItSpace: 8 logical tiles, 8 physical tiles
			Total Size: 102400 [D0, [7 x 12800, 12800]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [7 x 12800, 12800]]
		Tile0: [0, 12800, 12800], Tile1: [12800, 12800, 12800], Tile2; [25600, 12800, 12800]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 256 [D0, [7 x 32, 32]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [7 x 32, 32]]
		Tile0: [0, 256, 256], Tile1: [0, 256, 256], Tile2; [0, 256, 256]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 64 [D0, [7 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [7 x 8, 8]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 64 [D0, [7 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [7 x 8, 8]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 8 logical tiles, 1 physical tiles
			Total Size: 64 [D0, [7 x 8, 8]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [7 x 8, 8]]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (mnist_L1_Memory+0);
	KerArg0->InDim = (unsigned short int) (1600);
	KerArg0->TotalInDim = (unsigned short int) (1600);
	KerArg0->OutDim = (unsigned short int) (8);
	KerArg0->Infos = (signed char *__restrict__) (mnist_L1_Memory+27648);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0), 1600, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1600+0), 12800, 0, &DmaR_Evt2);
	_N_Filter=0;
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+27200), 256, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+27520), 64, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+27584), 64, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+27648), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	for (D0Ind=0; D0Ind<8; D0Ind++, D0Ind_Total++) { /* Iteration on D0 */
		int D0Ind_Last = (D0Ind==7), D0Ind_NextLast = ((D0Ind+1)==7);
		/*================================= Prepare Tiles ===================================*/
		_SN_Filter = 0;
		if (!(D0Ind_Last)) {
			_N_Filter = _N_Filter + (12800); _SN_Filter = (12800); 
		}
		/*============================= End Prepare Tiles ===================================*/
		/*================================= Read Tiles ======================================*/
		AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
		if (_SN_Filter) {
			AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+_N_Filter), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+1600+12800*((D0Ind_Total+1)%2)),
					_SN_Filter, 0, &DmaR_Evt2);
		}
		/*============================= End Read Tiles ======================================*/
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			KerArg0->Weights = (signed char * __restrict__) (mnist_L1_Memory+1600+12800*((D0Ind_Total)%2));
			KerArg0->Bias = (void * __restrict__) (mnist_L1_Memory+27200+((D0Ind)*32));
			KerArg0->Out = (void * __restrict__) (mnist_L1_Memory+27456+((D0Ind)*8));
			KerArg0->Scale = (unsigned char *__restrict__) (mnist_L1_Memory+27520+((D0Ind)*8));
			KerArg0->ScaleN = (unsigned char *__restrict__) (mnist_L1_Memory+27584+((D0Ind)*8));
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerFullFeatB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParLinearLayerFullFeatB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
		/*================================= Update Arg Pipeline =============================*/
		/*============================= End Update Arg Pipeline =============================*/
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+27456), 64, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S13_Op_FULLY_CONNECTED_0_6_fusion(
		signed char * __restrict__ In,
		signed char * __restrict__ Filter,
		int * __restrict__ Bias,
		signed char * __restrict__ Out,
		unsigned char * __restrict__ Scale,
		signed char * __restrict__ ScaleN,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 792 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaR_Evt3;
	AT_L2_EVENT DmaR_Evt4;
	AT_L2_EVENT DmaR_Evt5;
	AT_L2_EVENT DmaR_Evt6;
	AT_L2_EVENT DmaW_Evt1;
	KerLinear_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int D0Ind, D0Ind_Last;
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[D0 Dim: Init: 10, Tiled: 1][Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 64 [Tile0, 1:[1x1], 64]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 64]
		Tile0: [0, 64, 64], Tile1: [0, 64, 64], Tile2; [0, 64, 64]
	Ker Arg: Filter, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 640 [D0, [0 x 640, 640]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 640, 640]]
		Tile0: [0, 640, 640], Tile1: [0, 640, 640], Tile2; [0, 640, 640]
	Ker Arg: Bias, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 40 [D0, [0 x 40, 40]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 40, 40]]
		Tile0: [0, 40, 40], Tile1: [0, 40, 40], Tile2; [0, 40, 40]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10 [D0, [0 x 10, 10]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10, 10]]
		Tile0: [0, 10, 10], Tile1: [0, 10, 10], Tile2; [0, 10, 10]
	Ker Arg: Scale, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10 [D0, [0 x 10, 10]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10, 10]]
		Tile0: [0, 10, 10], Tile1: [0, 10, 10], Tile2; [0, 10, 10]
	Ker Arg: ScaleN, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10 [D0, [0 x 10, 10]]
		KerArgItSpace (User Kernel Iter Order):
			[D0, [0 x 10, 10]]
		Tile0: [0, 10, 10], Tile1: [0, 10, 10], Tile2; [0, 10, 10]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[1x1], 9]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x1], 9]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char * __restrict__) (mnist_L1_Memory+0);
	KerArg0->Weights = (signed char * __restrict__) (mnist_L1_Memory+64);
	KerArg0->Bias = (void * __restrict__) (mnist_L1_Memory+704);
	KerArg0->Out = (void * __restrict__) (mnist_L1_Memory+744);
	KerArg0->InDim = (unsigned short int) (64);
	KerArg0->TotalInDim = (unsigned short int) (64);
	KerArg0->OutDim = (unsigned short int) (10);
	KerArg0->Scale = (unsigned char *__restrict__) (mnist_L1_Memory+756);
	KerArg0->ScaleN = (unsigned char *__restrict__) (mnist_L1_Memory+768);
	KerArg0->Infos = (signed char *__restrict__) (mnist_L1_Memory+780);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0), 64, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Filter+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+64), 640, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Filter */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Bias+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+704), 40, 0, &DmaR_Evt3);
	AT_L2_WAIT(0, &DmaR_Evt3); /* Wait previous DMA read Bias */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Scale+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+756), 10, 0, &DmaR_Evt4);
	AT_L2_WAIT(0, &DmaR_Evt4); /* Wait previous DMA read Scale */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) ScaleN+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+768), 10, 0, &DmaR_Evt5);
	AT_L2_WAIT(0, &DmaR_Evt5); /* Wait previous DMA read ScaleN */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+780), 9, 0, &DmaR_Evt6);
	AT_L2_WAIT(0, &DmaR_Evt6); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on D0 */
		int D0Ind_Last = 1;
		{ /* Single iteration on Tile0 */
			int T0Ind_Last = 1;
			/*====================== Call Kernel LOC_LOOP =========================*/
			AT_FORK(gap_ncore(), (void *) KerParLinearLayerFullFeatB32_ReLU_SQ8, (void *) KerArg0);
			__CALL(KerParLinearLayerFullFeatB32_ReLU_SQ8, KerArg0);
		} /* End iteration on Tile0 */
	} /* End iteration on D0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+744), 10, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
void S14_SoftMax(
		signed char * __restrict__ In,
		short int * __restrict__ Out,
		signed char * __restrict__ Infos)

{
	/* Shared L1: 44 bytes, L2 buffer: 0 bytes */
	/* Local variables used by this kernel */
	AT_L2_EVENT DmaR_Evt1;
	AT_L2_EVENT DmaR_Evt2;
	AT_L2_EVENT DmaW_Evt1;
	KerSoftMax_SQ8_T S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Iteration space related variables */
	int T0Ind, T0Ind_Last;
	/* User kernel arguments related variables */
	/*============================= Ker Arg Iter Spaces =========================================
	User Kernel Iteration Space:
		[Tile0 Dim: 1]
	Ker Arg: In, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 10 [Tile0, 1:[1x10], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x10], 1]
		Tile0: [0, 10, 10], Tile1: [0, 10, 10], Tile2; [0, 10, 10]
	Ker Arg: Out, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 20 [Tile0, 1:[1x10], 2]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[1x10], 2]
		Tile0: [0, 20, 20], Tile1: [0, 20, 20], Tile2; [0, 20, 20]
	Ker Arg: Infos, Tiled Space: Buffer
		Min Pipe Depth: 0, Max Pipe Depth: 0
		KerArgItSpace: 1 logical tiles, 1 physical tiles
			Total Size: 9 [Tile0, 1:[9x1], 1]
		KerArgItSpace (User Kernel Iter Order):
			[Tile0, 1:[9x1], 1]
		Tile0: [0, 9, 9], Tile1: [0, 9, 9], Tile2; [0, 9, 9]
	======================== End Ker Arg Iter Spaces =========================================*/
	/*=========================== Call Kernel, Invariant assignment =====================*/
	KerArg0->In = (signed char *__restrict__) (mnist_L1_Memory+0);
	KerArg0->Feat = (unsigned short int) (1);
	KerArg0->N = (unsigned short int) (10);
	KerArg0->Out = (short int *__restrict__) (mnist_L1_Memory+12);
	KerArg0->Infos = (signed char *__restrict__) (mnist_L1_Memory+32);
	/*================================= Read Tiles Prolog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) In+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+0), 10, 0, &DmaR_Evt1);
	AT_L2_WAIT(0, &DmaR_Evt1); /* Wait previous DMA read In */
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Infos+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+32), 9, 0, &DmaR_Evt2);
	AT_L2_WAIT(0, &DmaR_Evt2); /* Wait previous DMA read Infos */
	/*============================= End Read Tiles Prolog ===============================*/
	{ /* Single iteration on Tile0 */
		int T0Ind_Last = 1;
		/*====================== Call Kernel LOC_LOOP =========================*/
		KerArg0->Norm = (unsigned short int) (((char *)(mnist_L1_Memory+32))[0]);
		AT_FORK(gap_ncore(), (void *) KerParSoftMax_SQ8, (void *) KerArg0);
		__CALL(KerParSoftMax_SQ8, KerArg0);
	} /* End iteration on Tile0 */
	/*================================ Write Tiles Epilog ===============================*/
	AT_L2_COPY(0, ((AT_L2_EXT_ADDR_TYPE) Out+0), ((AT_L2_INT_ADDR_TYPE) mnist_L1_Memory+12), 20, 1, &DmaW_Evt1);
	AT_L2_WAIT(0, &DmaW_Evt1); /* Wait DMA write Out */
	/*============================ End Write Tiles Epilog ===============================*/
}
int mnistCNN_Construct()

{
	AT_HYPERFLASH_FS_FC_EVENT UchanHF1;
	AT_HYPERFLASH_FS_CONF_T HyperFlashConf;
	int Error;
	AT_HYPERFLASH_FS_CONF_INIT(&HyperFlashConf, AT_MEM_L3_HFLASH, 0);
	AT_HYPERFLASH_FS_OPEN(&HyperFlash, &HyperFlashConf, "mnist_L3_Flash_Const.dat", &Error);
	if (Error) return 1;
	mnist_L2_Memory = (AT_L2_POINTER) AT_L2_ALLOC(0, 129564);
	if (mnist_L2_Memory == 0) return 3;
	mnist_L1_Memory = (AT_L1_POINTER) AT_L1_ALLOC(0, 44876);
	if (mnist_L1_Memory == 0) return 4;
	/* Moving Sequentialconv2dconv2d, size 800 from HyperFlash at 120832 to (size 800) L2 at 120832..121631 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 120832), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 120832), 800, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Conv2dbias, size 128 from HyperFlash at 122784 to (size 128) L2 at 122784..122911 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 122784), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 122784), 128, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Mul_scale, size 32 from HyperFlash at 123208 to (size 32) L2 at 123208..123239 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123208), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123208), 32, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Mul_shift, size 32 from HyperFlash at 123240 to (size 32) L2 at 123240..123271 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123240), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123240), 32, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S4_Infos, size 9 from HyperFlash at 123272 to (size 9) L2 at 123272..123280 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123272), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123272), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Sequentialconv2d_1conv2d, size 18432 from HyperFlash at 102400 to (size 18432) L2 at 102400..120831 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 102400), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 102400), 18432, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Conv2d_1bias, size 256 from HyperFlash at 122272 to (size 256) L2 at 122272..122527 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 122272), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 122272), 256, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Mul_scale, size 64 from HyperFlash at 122912 to (size 64) L2 at 122912..122975 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 122912), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 122912), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Mul_shift, size 64 from HyperFlash at 122976 to (size 64) L2 at 122976..123039 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 122976), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 122976), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S7_Infos, size 9 from HyperFlash at 123284 to (size 9) L2 at 123284..123292 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123284), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123284), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Sequentialdensematmul, size 102400 from HyperFlash at 0 to (size 102400) L2 at 0..102399 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 0), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 0), 102400, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Densebias, size 256 from HyperFlash at 122528 to (size 256) L2 at 122528..122783 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 122528), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 122528), 256, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Mul_scale, size 64 from HyperFlash at 123040 to (size 64) L2 at 123040..123103 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123040), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123040), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Mul_shift, size 64 from HyperFlash at 123104 to (size 64) L2 at 123104..123167 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123104), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123104), 64, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S10_Infos, size 9 from HyperFlash at 123296 to (size 9) L2 at 123296..123304 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123296), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123296), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Sequentialdense_1matmul, size 640 from HyperFlash at 121632 to (size 640) L2 at 121632..122271 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 121632), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 121632), 640, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving Dense_1bias, size 40 from HyperFlash at 123168 to (size 40) L2 at 123168..123207 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123168), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123168), 40, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S13_Mul_scale, size 10 from HyperFlash at 123308 to (size 10) L2 at 123308..123317 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123308), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123308), 10, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S13_Mul_shift, size 10 from HyperFlash at 123320 to (size 10) L2 at 123320..123329 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123320), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123320), 10, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S13_Infos, size 9 from HyperFlash at 123332 to (size 9) L2 at 123332..123340 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123332), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123332), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	/* Moving S14_Infos, size 9 from HyperFlash at 123344 to (size 9) L2 at 123344..123352 */
	AT_HYPERFLASH_FS_FC_COPY(&HyperFlash, ((AT_HYPERFLASH_FS_EXT_ADDR_TYPE) mnist_L3_Flash + 123344), ((AT_HYPERFLASH_FS_INT_ADDR_TYPE) mnist_L2_Memory + 123344), 9, 0, &UchanHF1);
	AT_HYPERFLASH_FS_FC_WAIT(&HyperFlash, &UchanHF1);
	return 0;
}
int mnistCNN_Destruct()

{
	AT_L2_FREE(0, mnist_L2_Memory, 129564);
	AT_L1_FREE(0, mnist_L1_Memory, 44876);
	AT_HYPERFLASH_FS_CLOSE(&HyperFlash);
	return 0;
}
int mnistCNN(
		unsigned char * __restrict__ Input_1,
		signed short * __restrict__ Output_1)

{
	S1_Op_input_1_formatter(
		((unsigned char * __restrict__) Input_1), /* In */
		((signed char * __restrict__) (mnist_L2_Memory+123356)) /* Out */
	);
	S4_Conv2d_32x1x5x5_MaxPool_2x2_Relu(
		((signed char * __restrict__) (mnist_L2_Memory+123356)), /* In */
		((signed char * __restrict__) (mnist_L2_Memory+120832)), /* Filter */
		((signed int * __restrict__) (mnist_L2_Memory+122784)), /* Bias */
		((signed char * __restrict__) (mnist_L2_Memory+124956)), /* Out */
		((unsigned char * __restrict__) (mnist_L2_Memory+123208)), /* Scale */
		((signed char * __restrict__) (mnist_L2_Memory+123240)), /* ScaleN */
		((signed char * __restrict__) (mnist_L2_Memory+123272)) /* Infos */
	);
	S7_Conv2d_64x32x3x3_MaxPool_2x2_Relu(
		((signed char * __restrict__) (mnist_L2_Memory+124956)), /* In */
		((signed char * __restrict__) (mnist_L2_Memory+102400)), /* Filter */
		((signed int * __restrict__) (mnist_L2_Memory+122272)), /* Bias */
		((signed char * __restrict__) (mnist_L2_Memory+123356)), /* Out */
		((unsigned char * __restrict__) (mnist_L2_Memory+122912)), /* Scale */
		((signed char * __restrict__) (mnist_L2_Memory+122976)), /* ScaleN */
		((signed char * __restrict__) (mnist_L2_Memory+123284)) /* Infos */
	);
	S10_Op_FULLY_CONNECTED_0_5_fusion(
		((signed char * __restrict__) (mnist_L2_Memory+123356)), /* In */
		((signed char * __restrict__) (mnist_L2_Memory+0)), /* Filter */
		((signed int * __restrict__) (mnist_L2_Memory+122528)), /* Bias */
		((signed char * __restrict__) (mnist_L2_Memory+124956)), /* Out */
		((unsigned char * __restrict__) (mnist_L2_Memory+123040)), /* Scale */
		((signed char * __restrict__) (mnist_L2_Memory+123104)), /* ScaleN */
		((signed char * __restrict__) (mnist_L2_Memory+123296)) /* Infos */
	);
	S13_Op_FULLY_CONNECTED_0_6_fusion(
		((signed char * __restrict__) (mnist_L2_Memory+124956)), /* In */
		((signed char * __restrict__) (mnist_L2_Memory+121632)), /* Filter */
		((signed int * __restrict__) (mnist_L2_Memory+123168)), /* Bias */
		((signed char * __restrict__) (mnist_L2_Memory+123356)), /* Out */
		((unsigned char * __restrict__) (mnist_L2_Memory+123308)), /* Scale */
		((signed char * __restrict__) (mnist_L2_Memory+123320)), /* ScaleN */
		((signed char * __restrict__) (mnist_L2_Memory+123332)) /* Infos */
	);
	S14_SoftMax(
		((signed char * __restrict__) (mnist_L2_Memory+123356)), /* In */
		((signed short * __restrict__) Output_1), /* Out */
		((signed char * __restrict__) (mnist_L2_Memory+123344)) /* Infos */
	);
	return 0;
}
