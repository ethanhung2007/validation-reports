# Cortex-A76 approximation for gem5 SE mode validation
# Based on O3_ARM_v7a, tuned to A76 Software Optimization Guide values

from m5.objects import *
from common.cores.arm.O3_ARM_v7a import (
    O3_ARM_v7a_Simple_Int,
    O3_ARM_v7a_Complex_Int,
    O3_ARM_v7a_FP,
    O3_ARM_v7a_Store,
)

# ---- Functional Units ----
# A76 has 2 load pipes vs v7a's 1
class A76_Load(FUDesc):
    opList = [
        OpDesc(opClass='MemRead',      opLat=4),
        OpDesc(opClass='FloatMemRead', opLat=4),
    ]
    count = 2

class A76_Store(FUDesc):
    opList = [
        OpDesc(opClass='MemWrite',      opLat=1),
        OpDesc(opClass='FloatMemWrite', opLat=1),
    ]
    count = 1

class A76_FUP(FUPool):
    FUList = [
        O3_ARM_v7a_Simple_Int(),   # 2x ALU, lat=1
        O3_ARM_v7a_Complex_Int(),  # 1x MUL/DIV
        A76_Load(),                # 2x load
        A76_Store(),               # 1x store
        O3_ARM_v7a_FP(),           # 2x FP/SIMD — latencies match A76 SOG
    ]

# ---- Branch Predictor ----
class A76_BP(BranchPredictor):
    conditionalBranchPred = TAGE_SC_L_64KB()
    btb = SimpleBTB(numEntries=4096)
    ras = ReturnAddrStack(numEntries=32)
    instShiftAmt = 2

# ---- Issue Queue ----
class A76_IQ(IQUnit):
    fuPool = A76_FUP()
    numEntries = 60

# ---- Core ----
class CortexA76(ArmO3CPU):
    # Frontend — 4-wide
    fetchWidth        = 4
    fetchBufferSize   = 64
    fetchToDecodeDelay = 3
    decodeWidth       = 4
    decodeToRenameDelay = 2
    renameWidth       = 4
    renameToIEWDelay  = 1

    # Backend — up to 8-wide issue
    dispatchWidth     = 8
    issueWidth        = 8
    wbWidth           = 8
    commitWidth       = 4
    squashWidth       = 8

    # Pipeline stage delays (keep v7a defaults)
    decodeToFetchDelay  = 1
    renameToFetchDelay  = 1
    iewToFetchDelay     = 1
    commitToFetchDelay  = 1
    renameToDecodeDelay = 1
    iewToDecodeDelay    = 1
    commitToDecodeDelay = 1
    iewToRenameDelay    = 1
    commitToRenameDelay = 1
    commitToIEWDelay    = 1
    issueToExecuteDelay = 1
    iewToCommitDelay    = 1
    renameToROBDelay    = 1

    # Buffers — A76 TRM values
    numROBEntries     = 128
    LQEntries         = 68
    SQEntries         = 72
    LSQDepCheckShift  = 0

    # Physical register file
    numPhysIntRegs    = 180
    numPhysFloatRegs  = 192
    numPhysVecRegs    = 128

    switched_out = False
    branchPred   = A76_BP()
    instQueues   = A76_IQ()

# ---- L1 Instruction Cache (64KiB, 4-way) ----
class A76_ICache(Cache):
    size           = '64KiB'
    assoc          = 4
    tag_latency    = 1
    data_latency   = 1
    response_latency = 1
    mshrs          = 16
    tgts_per_mshr  = 8
    is_read_only   = True
    writeback_clean = True

# ---- L1 Data Cache (64KiB, 4-way) ----
class A76_DCache(Cache):
    size           = '64KiB'
    assoc          = 4
    tag_latency    = 4
    data_latency   = 4
    response_latency = 4
    mshrs          = 16
    tgts_per_mshr  = 8
    write_buffers  = 32
    writeback_clean = True

# ---- L2 Cache (512KiB private, 8-way) ----
class A76_L2(Cache):
    size           = '512KiB'
    assoc          = 8
    tag_latency    = 8
    data_latency   = 8
    response_latency = 8
    mshrs          = 32
    tgts_per_mshr  = 8
    write_buffers  = 16
    clusivity      = 'mostly_excl'
    prefetcher     = StridePrefetcher(degree=8, latency=1, prefetch_on_access=True)
    tags           = BaseSetAssoc()
    replacement_policy = RandomRP()
