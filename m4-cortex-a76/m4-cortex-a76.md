# M4 Cortex A76

The testing and validation is done using gem5 with the configuration mainly being from sources provided by ARM with a few estimated values due to ARM not fully releasing the full microarchitecture of the A76 core. 

## Configuration

```text
fetchWidth = 4
decodeWidth = 4
renameWidth = 4
dispatchWidth = 8
issueWidth = 8
commitWidth = 4
numROBEntries = 128
LQEntries = 68
SQEntries = 72
numPhysIntRegs = 180
numPhysFloatRegs = 192
numPhysVecRegs = 128
conditionalBranchPred = TAGE_SC_L_64KB()
btb = SimpleBTB(numEntries=4096)
ras = ReturnAddrStack(numEntries=32)
L1I: 64KiB, 4-way
L1D: 64KiB, 4-way
L2:  512KiB, 8-way (private per core)
L3: 4MiB
```

## Checks

The checks were written and ran through gem5 for results then compared with recorded data.

1. residual
2. rmsnorm
3. rope_apply
4. swiglu
5. sampling_argmax
6. softmax (kv128, 512, 1024)

