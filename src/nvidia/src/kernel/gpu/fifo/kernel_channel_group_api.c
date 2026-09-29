/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "kernel/gpu/fifo/kernel_channel_group_api.h"

#include "kernel/core/locks.h"
#include "kernel/gpu/fifo/kernel_channel_group.h"
#include "kernel/gpu/fifo/kernel_fifo.h"
#include "kernel/gpu/mem_mgr/mem_mgr.h"
#include "kernel/gpu/gr/kernel_graphics.h"
#include "kernel/gpu/falcon/kernel_falcon.h"
#include "kernel/gpu/rc/kernel_rc.h"
#include "platform/sli/sli.h"

#include "kernel/gpu/conf_compute/conf_compute.h"

#include "class/cl0090.h" // KERNEL_GRAPHICS_CONTEXT
#include "class/cl9067.h" // FERMI_CONTEXT_SHARE_A

#include "libraries/utils/nvprintf.h"
#include "gpu/gpu.h"
#include "gpu/device/device.h"
#include "kernel/gpu/mig_mgr/kernel_mig_manager.h"
#include "gpu/mem_mgr/vaspace_api.h"
#include "vgpu/rpc.h"
#include "rmapi/rs_utils.h"
#include "ctrl/ctrl2080/ctrl2080fifo.h" // GHOST 0i: FIFO_DISABLE_CHANNELS (detach+preempt)
#include "ctrl/ctrla06f/ctrla06fgpfifo.h" // GHOST: RESTART_RUNLIST (force preempt)
#include "kernel/rmapi/rmapi.h"        // GHOST: g_resServ (client lookup)
#include "kernel/gpu/fifo/kernel_channel.h" // GHOST: CliGetKernelChannel, USERD
#include "containers/eheap_old.h"

#include "kernel/os/os.h"

// GHOST EXPERIMENT (0e): deferred re-probe, defined in kernel_ctxshare.c.
extern void ghostReprobeDeferred_GHOST(OBJGPU *pGpu);
// GHOST EXPERIMENT (0g): per-process tenant index, defined in kernel_ctxshare.c.
extern NvU32 ghostTenantIndex_GHOST(void);

// ============================================================================
// GHOST GPU SCHEDULER: dynamic runlist detach/attach driven from userspace.
//
// A trusted host component (runsc gpu-scheduler) time-slices tenants by writing
// "detach <pid>" / "attach <pid>" to /proc/driver/nvidia/gpusched. This is the
// mechanism half: record each tenant's channels as they are scheduled, and on
// command issue FifoDisableChannels with a forced preempt (detach) or re-enable
// (attach). The control originates at kernel privilege inside an RM work item,
// so it holds the API+GPU locks and can RPC to GSP -- exactly Ghost's
// DetachTSG/AttachTSG. The policy (who runs when, work-conservation) lives in
// the userspace scheduler; this code only enforces its decisions.
// ============================================================================
#define GHOST_MAX_GROUPS 256
typedef struct
{
    NvU32     pid;        // owning process (the KVM Sentry, one per sandbox)
    OBJGPU   *pGpu;       // the GPU this group runs on; its RPCs must go there
    NvHandle  hClient;
    NvHandle  hGroup;     // the channel-group (TSG) API handle, for SET_TIMESLICE
    NvHandle  hChannels[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
    NvU32     nCh;
    NvBool    valid;
    // Activity signal: the last GP_PUT seen on each channel, and whether any
    // advanced on the most recent poll. This is the trusted, doorbell-aware
    // "is this sandbox submitting work" signal -- read from the channel's USERD
    // in the driver, so a tenant cannot lie about it the way it could about an
    // in-container kernel counter, and it sees doorbell submission that fault
    // counts and nvidia-smi miss or mis-time.
    NvU32     lastGpPut[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
    NvBool    active;
} GHOST_GROUP;

// NV_RAMUSERD_GP_GET / GP_PUT live at words 34/35 of the channel's USERD
// (GA100/Maxwell layout), i.e. byte offsets 136 and 140. A channel has pending,
// unconsumed work exactly when PUT != GET -- that is the "wants to run" signal.
// (PUT-advanced-since-last-poll is wrong: a time-sliced tenant waiting its turn
// has already pushed its work, so PUT is static though GET has not caught up.)
#define GHOST_USERD_GP_GET_OFFSET 136
#define GHOST_USERD_GP_PUT_OFFSET 140
static GHOST_GROUP g_ghostGroups[GHOST_MAX_GROUPS];
static NvU32       g_ghostGroupCount = 0;
static OBJGPU     *g_ghostGpu = NULL;
static NvBool      g_ghostFullWarned = NV_FALSE;

// ghostSchedRecord_GHOST: remember a channel group and its channels, keyed by
// the calling process, so a later detach/attach can find them. Called from the
// GPFIFO_SCHEDULE hook while the group is being enabled.
void
ghostSchedRecord_GHOST(OBJGPU *pGpu, NvU32 pid, NvHandle hClient,
                       NvHandle hGroup, KernelChannelGroup *pKcg)
{
    NvU32 i, slot;
    CHANNEL_NODE *pNode;

    if (pKcg == NULL || pKcg->pChanList == NULL)
        return;
    g_ghostGpu = pGpu;

    // Reuse a slot for the same (pid, hClient, hGroup) if we already have one;
    // otherwise take the first slot ghostSchedForget_GHOST released, and only
    // then grow the table.
    slot = g_ghostGroupCount;
    {
        NvU32 freeSlot = GHOST_MAX_GROUPS;
        for (i = 0; i < g_ghostGroupCount; i++)
        {
            if (!g_ghostGroups[i].valid)
            {
                if (freeSlot == GHOST_MAX_GROUPS)
                    freeSlot = i;
                continue;
            }
            if (g_ghostGroups[i].pid == pid &&
                g_ghostGroups[i].hClient == hClient &&
                g_ghostGroups[i].hGroup == hGroup)
            {
                freeSlot = GHOST_MAX_GROUPS;
                slot = i;
                break;
            }
        }
        if (freeSlot != GHOST_MAX_GROUPS)
            slot = freeSlot;
    }
    if (slot >= GHOST_MAX_GROUPS)
    {
        // An untracked group is one the scheduler can never detach, so the
        // tenant runs unthrottled: this is a fail-open, and must be loud.
        if (!g_ghostFullWarned)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "GHOST sched: group table full (%u); pid %u group 0x%08x "
                      "is NOT tracked and will not be throttled\n",
                      GHOST_MAX_GROUPS, pid, hGroup);
            g_ghostFullWarned = NV_TRUE;
        }
        return;
    }

    portMemSet(&g_ghostGroups[slot], 0, sizeof(g_ghostGroups[slot]));
    g_ghostGroups[slot].pid     = pid;
    g_ghostGroups[slot].pGpu    = pGpu;
    g_ghostGroups[slot].hClient = hClient;
    g_ghostGroups[slot].hGroup  = hGroup;
    for (pNode = pKcg->pChanList->pHead;
         pNode && (g_ghostGroups[slot].nCh < NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES);
         pNode = pNode->pNext)
    {
        if (pNode->pKernelChannel == NULL)
            continue;
        g_ghostGroups[slot].hChannels[g_ghostGroups[slot].nCh++] =
            RES_GET_HANDLE(pNode->pKernelChannel);
    }
    g_ghostGroups[slot].valid = NV_TRUE;
    if (slot == g_ghostGroupCount)
        g_ghostGroupCount++;
}

// ghostSchedForget_GHOST: release the slot recorded for (hClient, hGroup).
// Called from the channel-group API destructor, which every teardown path goes
// through -- an explicit free, and the client free when a process (a sandbox)
// exits. Without it slots are never reclaimed: the table fills after a few
// hundred channel groups (a multi-GPU NCCL job alone takes ~50), and from then
// on every new tenant goes untracked and unthrottled. Runs under the same RM
// locks as ghostSchedRecord_GHOST and the work item.
static void
ghostSchedForget_GHOST(NvHandle hClient, NvHandle hGroup)
{
    NvU32 i;

    for (i = 0; i < g_ghostGroupCount; i++)
    {
        if (g_ghostGroups[i].valid &&
            g_ghostGroups[i].hClient == hClient &&
            g_ghostGroups[i].hGroup == hGroup)
        {
            g_ghostGroups[i].valid = NV_FALSE;
            g_ghostGroups[i].nCh = 0;
            g_ghostFullWarned = NV_FALSE;
        }
    }
    while ((g_ghostGroupCount > 0) && !g_ghostGroups[g_ghostGroupCount - 1].valid)
        g_ghostGroupCount--;
}

// Parameters carried to the work item.
typedef struct { NvU32 pid; NvU32 action; NvU32 arg; } GHOST_SCHED_WI;
// action: 0=attach, 1=detach(+RESTART_RUNLIST), 2=set-timeslice arg=us (+RESTART)

// ghostRestartRunlist: force GSP to expire the current timeslice and restart the
// runlist the channel belongs to -- a manual, privileged preemption. This is the
// companion action the earlier detach/timeslice hooks were missing: they changed
// per-object state without forcing GSP to re-evaluate. Issued on the channel
// object (A06F) via the owning client from a kernel-priv work item.
static void
ghostRestartRunlist(OBJGPU *pGpu, NvHandle hClient, NvHandle hChannel)
{
    NVA06F_CTRL_RESTART_RUNLIST_PARAMS rr;
    NV_STATUS st = NV_OK;
    portMemSet(&rr, 0, sizeof(rr));
    rr.bForceRestart = NV_TRUE;
    rr.bBypassWait   = NV_FALSE;
    NV_RM_RPC_CONTROL(pGpu, hClient, hChannel,
                      NVA06F_CTRL_CMD_RESTART_RUNLIST, &rr, sizeof(rr), st);
    NV_PRINTF(LEVEL_ERROR,
              "GHOST sched: RESTART_RUNLIST chan 0x%08x -> 0x%x\n", hChannel, st);
}

static void
ghostSchedWorkItem(NvU32 gpuInstance, void *pParams)
{
    GHOST_SCHED_WI *pWi = (GHOST_SCHED_WI *)pParams;
    NvU32 i, c;

    if (g_ghostGpu == NULL || pWi == NULL)
        goto done;

    for (i = 0; i < g_ghostGroupCount; i++)
    {
        NV_STATUS st = NV_OK;
        //
        // Every control below must reach the GPU that owns the group: a
        // group's handles mean nothing to another GPU's GSP, which answers
        // NV_ERR_OBJECT_NOT_FOUND. The work item holds every GPU's lock
        // (bLockGpus => GPU_LOCK_GRP_ALL), so any of them may be addressed
        // here, not just the one it was queued on.
        //
        OBJGPU *pGpu = g_ghostGroups[i].pGpu;

        if (!g_ghostGroups[i].valid || g_ghostGroups[i].nCh == 0 || pGpu == NULL)
            continue;
        if (pWi->action != 3 && g_ghostGroups[i].pid != pWi->pid)
            continue;

        if (pWi->action == 3)
        {
            // Poll GP_PUT on each of the tenant's channels and flag the group
            // active if any advanced since the last poll. Runs under the RM
            // locks the work item holds, so the handle lookups are safe against
            // a channel being freed concurrently.
            MemoryManager *pMM = GPU_GET_MEMORY_MANAGER(pGpu);
            RsClient *pClient = NULL;
            NvBool anyAdvanced = NV_FALSE;
            if (serverGetClientUnderLock(&g_resServ, g_ghostGroups[i].hClient,
                                         &pClient) == NV_OK && pClient != NULL)
            {
                for (c = 0; c < g_ghostGroups[i].nCh; c++)
                {
                    KernelChannel *pKC = NULL;
                    MEMORY_DESCRIPTOR *pUserd;
                    NvU8 *pMap;
                    NvU32 gpput;
                    if (CliGetKernelChannel(pClient, g_ghostGroups[i].hChannels[c],
                                            &pKC) != NV_OK || pKC == NULL)
                        continue;
                    pUserd = pKC->pUserdSubDeviceMemDesc[0];
                    if (pUserd == NULL)
                        continue;
                    NvU32 gpget;
                    pMap = memmgrMemDescBeginTransfer(pMM, pUserd,
                                                      TRANSFER_FLAGS_SHADOW_ALLOC);
                    if (pMap == NULL)
                        continue;
                    gpget = *(volatile NvU32 *)(pMap + GHOST_USERD_GP_GET_OFFSET);
                    gpput = *(volatile NvU32 *)(pMap + GHOST_USERD_GP_PUT_OFFSET);
                    memmgrMemDescEndTransfer(pMM, pUserd, TRANSFER_FLAGS_SHADOW_ALLOC);
                    // Pending work (PUT != GET) OR fresh submission since the
                    // last poll (PUT advanced) both mean the tenant wants the GPU.
                    if (gpput != gpget || gpput != g_ghostGroups[i].lastGpPut[c])
                        anyAdvanced = NV_TRUE;
                    g_ghostGroups[i].lastGpPut[c] = gpput;
                }
            }
            g_ghostGroups[i].active = anyAdvanced;
            continue;
        }

        if (pWi->action == 2)
        {
            // SET_TIMESLICE on the TSG, then RESTART_RUNLIST to force GSP to
            // re-read the runlist and preempt now.
            NVA06C_CTRL_TIMESLICE_PARAMS ts;
            portMemSet(&ts, 0, sizeof(ts));
            ts.timesliceUs = pWi->arg;
            NV_RM_RPC_CONTROL(pGpu, g_ghostGroups[i].hClient,
                              g_ghostGroups[i].hGroup,
                              NVA06C_CTRL_CMD_SET_TIMESLICE, &ts, sizeof(ts), st);
            NV_PRINTF(LEVEL_ERROR,
                      "GHOST sched: pid %u SET_TIMESLICE grp 0x%08x = %u us -> 0x%x\n",
                      pWi->pid, g_ghostGroups[i].hGroup, pWi->arg, st);
            for (c = 0; c < g_ghostGroups[i].nCh; c++)
                ghostRestartRunlist(pGpu, g_ghostGroups[i].hClient,
                                    g_ghostGroups[i].hChannels[c]);
            continue;
        }

        // attach / detach via DISABLE_CHANNELS
        {
            NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS *pDis =
                portMemAllocNonPaged(sizeof(*pDis));
            if (pDis == NULL)
                continue;
            portMemSet(pDis, 0, sizeof(*pDis));
            pDis->bDisable = (pWi->action == 0) ? NV_FALSE : NV_TRUE;
            pDis->bOnlyDisableScheduling = NV_FALSE;
            pDis->bRewindGpPut = NV_FALSE;
            pDis->pRunlistPreemptEvent = NULL;
            pDis->numChannels = g_ghostGroups[i].nCh;
            for (c = 0; c < g_ghostGroups[i].nCh; c++)
            {
                pDis->hClientList[c]  = g_ghostGroups[i].hClient;
                pDis->hChannelList[c] = g_ghostGroups[i].hChannels[c];
            }
            NV_RM_RPC_CONTROL(pGpu, pGpu->hInternalClient, pGpu->hInternalSubdevice,
                              NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS,
                              pDis, sizeof(*pDis), st);
            NV_PRINTF(LEVEL_ERROR,
                      "GHOST sched: pid %u %s %u channels -> 0x%x\n",
                      pWi->pid, (pWi->action == 0) ? "ATTACH" : "DETACH",
                      g_ghostGroups[i].nCh, st);
            portMemFree(pDis);
        }

        // On detach, force a runlist restart so the disable takes effect on
        // already-running work instead of only at the next natural idle.
        if (pWi->action == 1)
        {
            for (c = 0; c < g_ghostGroups[i].nCh; c++)
                ghostRestartRunlist(pGpu, g_ghostGroups[i].hClient,
                                    g_ghostGroups[i].hChannels[c]);
        }
    }

done:
    portMemFree(pParams);
}

int ghostSchedFormatActive_GHOST(char *buf, int cap); // defined below

void
ghostSchedQueue_GHOST(NvU32 pid, NvU32 action, NvU32 arg)
{
    GHOST_SCHED_WI *pWi;
    OsQueueWorkItemFlags flags;
    // Reference the OS-layer-only formatter so the RM-core --gc-sections link
    // does not garbage-collect it (nothing else inside RM core calls it). The
    // guard is never true; the compiler cannot prove that across the parameter,
    // so the call -- and thus the symbol reference -- survives.
    if (action == 0xDEADBEEF)
    {
        char tmp[1];
        (void)ghostSchedFormatActive_GHOST(tmp, 0);
    }

    if (g_ghostGpu == NULL)
        return;
    pWi = portMemAllocNonPaged(sizeof(*pWi));
    if (pWi == NULL)
        return;
    pWi->pid = pid;
    pWi->action = action;
    pWi->arg = arg;

    portMemSet(&flags, 0, sizeof(flags));
    flags.bRequiresGpu = NV_TRUE;
    flags.apiLock = WORKITEM_FLAGS_API_LOCK_READ_WRITE;
    flags.bLockGpus = NV_TRUE;
    flags.bDontFreeParams = NV_TRUE; // the callback frees pParams itself
    if (osQueueWorkItem(g_ghostGpu, ghostSchedWorkItem, pWi, flags) != NV_OK)
        portMemFree(pWi);
}

// ghostSchedFormatActive_GHOST: write "pid <p> active <0|1>" for each distinct
// tenant into buf (from the cached poll result), returning the length. Called
// from the procfs read handler; reads only cached flags, no locks.
int
ghostSchedFormatActive_GHOST(char *buf, int cap)
{
    NvU32 i, j;
    int n = 0;
    NvU32 seen[GHOST_MAX_GROUPS];
    NvU32 nSeen = 0;

    for (i = 0; i < g_ghostGroupCount && nSeen < GHOST_MAX_GROUPS; i++)
    {
        NvBool dup = NV_FALSE;
        NvBool act;
        if (!g_ghostGroups[i].valid)
            continue;
        for (j = 0; j < nSeen; j++)
            if (seen[j] == g_ghostGroups[i].pid) { dup = NV_TRUE; break; }
        if (dup)
            continue;
        seen[nSeen++] = g_ghostGroups[i].pid;
        // active if any group of this pid is active; nTsg is how many groups it
        // owns, which with a fixed per-TSG quantum is proportional to the share
        // it actually takes -- the scheduler charges credit by it.
        act = NV_FALSE;
        NvU32 nTsg = 0;
        for (j = 0; j < g_ghostGroupCount; j++)
            if (g_ghostGroups[j].valid && g_ghostGroups[j].nCh > 0 &&
                g_ghostGroups[j].pid == g_ghostGroups[i].pid)
            {
                nTsg++;
                if (g_ghostGroups[j].active)
                    act = NV_TRUE;
            }
        if (n < cap)
            n += nvDbgSnprintf(buf + n, cap - n, "pid %u active %d tsgs %u\n",
                               g_ghostGroups[i].pid, act ? 1 : 0, nTsg);
    }
    return n;
}


NV_STATUS
kchangrpapiConstruct_IMPL
(
    KernelChannelGroupApi        *pKernelChannelGroupApi,
    CALL_CONTEXT                 *pCallContext,
    RS_RES_ALLOC_PARAMS_INTERNAL *pParams
)
{
    NvBool                                  bTsgAllocated       = NV_FALSE;
    RsResourceRef                          *pResourceRef        = pCallContext->pResourceRef;
    NV_STATUS                               rmStatus;
    OBJVASPACE                             *pVAS                = NULL;
    OBJGPU                                 *pGpu                = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    KernelMIGManager                       *pKernelMIGManager   = NULL;
    KernelFifo                             *pKernelFifo         = GPU_GET_KERNEL_FIFO(pGpu);
    NvHandle                                hVASpace            = NV01_NULL_OBJECT;
    Device                                 *pDevice             = NULL;
    NvU32                                   gfid                = GPU_GFID_PF;
    RsShared                               *pShared             = NULL;
    RsClient                               *pClient;
    NvBool                                  bLockAcquired       = NV_FALSE;
    Heap                                   *pHeap               = GPU_GET_HEAP(pGpu);
    NvBool                                  bMIGInUse           = NV_FALSE;
    CTX_BUF_INFO                           *bufInfoList         = NULL;
    NvU32                                   bufCount            = 0;
    NvBool                                  bReserveMem         = NV_FALSE;
    MIG_INSTANCE_REF                        ref;
    RM_API                                 *pRmApi              = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);
    KernelChannelGroup                     *pKernelChannelGroup = NULL;
    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS *pAllocParams        = NULL;
    RM_ENGINE_TYPE                          rmEngineType;


    NV_PRINTF(LEVEL_INFO,
              "hClient: 0x%x, hParent: 0x%x, hObject:0x%x, hClass: 0x%x\n",
              pParams->hClient, pParams->hParent, pParams->hResource,
              pParams->externalClassId);

    if (RS_IS_COPY_CTOR(pParams))
    {
        NV_ASSERT_OK_OR_GOTO(rmStatus,
                             rmDeviceGpuLocksAcquire(pGpu, GPUS_LOCK_FLAGS_NONE, RM_LOCK_MODULES_FIFO),
                             done);
        bLockAcquired = NV_TRUE;
        rmStatus = kchangrpapiCopyConstruct_IMPL(pKernelChannelGroupApi,
                                                 pCallContext, pParams);
        goto done;
    }

    //
    // Make sure this GPU is not already locked by this thread
    // Ideally this thread shouldn't have locked any GPU in the system but
    // checking this is sufficient as memory allocation from PMA requires
    // current GPU's lock not to be held
    //
    if (rmDeviceGpuLockIsOwner(pGpu->gpuInstance))
    {
        NV_PRINTF(LEVEL_ERROR, "TSG alloc should be called without acquiring GPU lock\n");
        NV_ASSERT_OR_RETURN(0, NV_ERR_INVALID_LOCK_STATE);
    }

    bufInfoList = portMemAllocNonPaged(NV_ENUM_SIZE(GR_CTX_BUFFER) * sizeof(*bufInfoList));
    if (bufInfoList == NULL)
    {
        return NV_ERR_NO_MEMORY;
    }

    // Acquire the lock *only after* PMA is done allocating.
    NV_ASSERT_OK_OR_GOTO(rmStatus,
                         rmDeviceGpuLocksAcquire(pGpu, GPUS_LOCK_FLAGS_NONE, RM_LOCK_MODULES_FIFO),
                         done);
    bLockAcquired = NV_TRUE;

    pAllocParams = pParams->pAllocParams;
    hVASpace     = pAllocParams->hVASpace;

    NV_ASSERT_OK_OR_GOTO(rmStatus,
        serverAllocShareWithHalspecParent(&g_resServ, classInfo(KernelChannelGroup),
                                          &pShared, staticCast(pGpu, Object)),
        failed);

    pKernelChannelGroup = dynamicCast(pShared, KernelChannelGroup);
    pKernelChannelGroupApi->pKernelChannelGroup = pKernelChannelGroup;

    if (!gpuIsClassSupported(pGpu, pResourceRef->externalClassId))
    {
        NV_PRINTF(LEVEL_ERROR, "class %x not supported\n",
                  pResourceRef->externalClassId);
        rmStatus = NV_ERR_NOT_SUPPORTED;
        goto failed;
    }

    pKernelChannelGroupApi->hVASpace = hVASpace;

    rmStatus = serverGetClientUnderLock(&g_resServ, pParams->hClient, &pClient);
    if (rmStatus != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "Invalid client handle!\n");
        rmStatus = NV_ERR_INVALID_ARGUMENT;
        goto failed;
    }

    rmStatus = deviceGetByHandle(pClient, pParams->hParent, &pDevice);
    if (rmStatus != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "Invalid parent/device handle!\n");
        rmStatus = NV_ERR_INVALID_ARGUMENT;
        goto failed;
    }

    pKernelMIGManager = GPU_GET_KERNEL_MIG_MANAGER(pGpu);
    bMIGInUse = IS_MIG_IN_USE(pGpu);

    rmEngineType = gpuGetRmEngineType(pAllocParams->engineType);

    if (kfifoIsPerRunlistChramSupportedInHw(pKernelFifo))
    {
        if (!RM_ENGINE_TYPE_IS_VALID(rmEngineType))
        {
            NV_PRINTF(LEVEL_NOTICE, "Valid engine Id must be specified while allocating TSGs or bare channels!\n");
            rmStatus = NV_ERR_INVALID_ARGUMENT;
            goto failed;
        }

        //
        // If we have a separate channel RAM for each runlist then we need
        // to determine runlistId from engineId passed by client. This
        // runlistId is used to associate all future channels in this TSG to
        // that runlist. Setting the engineType will cause the runlist
        // corresponding to that engine to be chosen.
        //
        pKernelChannelGroup->engineType = rmEngineType;
    }

    //
    // If MIG is enabled, client passes a logical engineId w.r.t its own GPU instance
    // we need to convert this logical Id to a physical engine Id as we use it
    // to set runlistId
    //
    if (bMIGInUse)
    {
        // Engine type must be valid for MIG
        NV_CHECK_OR_ELSE(LEVEL_NOTICE, RM_ENGINE_TYPE_IS_VALID(pKernelChannelGroup->engineType),
                         rmStatus = NV_ERR_INVALID_STATE; goto failed);

        NV_CHECK_OK_OR_GOTO(
            rmStatus,
            LEVEL_ERROR,
            kmigmgrGetInstanceRefFromDevice(pGpu, pKernelMIGManager,
                                            pDevice, &ref),
            failed);

        NV_CHECK_OK_OR_GOTO(
            rmStatus,
            LEVEL_ERROR,
            kmigmgrGetLocalToGlobalEngineType(pGpu, pKernelMIGManager, ref,
                                              rmEngineType,
                                              &rmEngineType),
            failed);

        // Rewrite the engineType with the global engine type
        pKernelChannelGroup->engineType = rmEngineType;
        pHeap = ref.pKernelMIGGpuInstance->pMemoryPartitionHeap;
    }
    else
    {
        // Only GR0 is allowed without MIG
        if ((RM_ENGINE_TYPE_IS_GR(rmEngineType)) && (rmEngineType != RM_ENGINE_TYPE_GR0))
        {
            rmStatus = NV_ERR_INVALID_ARGUMENT;
            goto failed;
        }
    }

    if((pDevice->vaMode != NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES) || (hVASpace != 0))
    {
        NV_ASSERT_OK_OR_GOTO(rmStatus,
            vaspaceGetByHandleOrDeviceDefault(pClient, pParams->hParent, hVASpace, &pVAS),
            failed);

        if (pVAS == NULL)
        {
            rmStatus = NV_ERR_INVALID_STATE;
            goto failed;
        }
    }


    // vGpu plugin context flag should only be set on host if context is plugin
    if (gpuIsSriovEnabled(pGpu))
    {
        pKernelChannelGroup->bIsCallingContextVgpuPlugin = pAllocParams->bIsCallingContextVgpuPlugin;
    }

    if (pKernelChannelGroup->bIsCallingContextVgpuPlugin)
        gfid = GPU_GFID_PF;
    else
    {
        NV_ASSERT_OK_OR_GOTO(rmStatus, vgpuGetCallingContextGfid(pGpu, &gfid), failed);
    }

    if (!RMCFG_FEATURE_PLATFORM_GSP)
    {
        RmClient *pRmClient = dynamicCast(pClient, RmClient);
        //
        // WAR for 4217716 - Force allocations made on behalf of internal clients to
        // RM reserved heap. This avoids a constant memory allocation from appearing
        // due to the ctxBufPool reservation out of PMA.
        //
        if ((pRmClient == NULL) || !serverIsClientInternal(&g_resServ, pParams->hClient))
        {
            NV_ASSERT_OK_OR_GOTO(rmStatus,
                ctxBufPoolInit(pGpu, pHeap, &pKernelChannelGroup->pCtxBufPool),
                failed);

            NV_ASSERT_OK_OR_GOTO(rmStatus,
                ctxBufPoolInit(pGpu, pHeap, &pKernelChannelGroup->pChannelBufPool),
                failed);
        }
        else
        {
            NV_PRINTF(LEVEL_INFO, "Skipping ctxBufPoolInit for RC watchdog\n");
        }
    }

    NV_ASSERT_OK_OR_GOTO(rmStatus,
                         kchangrpInit(pGpu, pKernelChannelGroup, pVAS, gfid),
                         failed);
    bTsgAllocated = NV_TRUE;

    pKernelChannelGroupApi->hLegacykCtxShareSync  = 0;
    pKernelChannelGroupApi->hLegacykCtxShareAsync = 0;

    if (hVASpace != 0)
    {
        RsResourceRef *pVASpaceRef;
        rmStatus = clientGetResourceRef(pCallContext->pClient, hVASpace, &pVASpaceRef);
        NV_ASSERT(rmStatus == NV_OK);
        if (rmStatus == NV_OK)
            refAddDependant(pVASpaceRef, pResourceRef);
    }

    pKernelChannelGroupApi->hErrorContext    = pAllocParams->hObjectError;
    pKernelChannelGroupApi->hEccErrorContext = pAllocParams->hObjectEccError;

    // Default interleave level
    NV_ASSERT_OK_OR_GOTO(
        rmStatus,
        kchangrpSetInterleaveLevel(pGpu, pKernelChannelGroup,
                                   NVA06C_CTRL_INTERLEAVE_LEVEL_MEDIUM),
        failed);

    ConfidentialCompute *pConfCompute = GPU_GET_CONF_COMPUTE(pGpu);
    MemoryManager *pMemoryManager = GPU_GET_MEMORY_MANAGER(pGpu);
    if ((pConfCompute != NULL) &&
        (pConfCompute->getProperty(pCC, PDB_PROP_CONFCOMPUTE_CC_FEATURE_ENABLED)))
    {
        // TODO: jira CONFCOMP-1621: replace this with actual flag for TSG alloc that skips scrub
        if ((pMemoryManager->bScrubChannelSetupInProgress) &&
            (pKernelChannelGroup->pChannelBufPool != NULL) &&
            (pKernelChannelGroup->pCtxBufPool != NULL))
        {
            if (pCallContext->secInfo.privLevel < RS_PRIV_LEVEL_KERNEL)
            {
                rmStatus = NV_ERR_INVALID_ARGUMENT;
                NV_PRINTF(LEVEL_ERROR, "Only kernel priv clients can skip scrubber\n");
                goto failed;
            }
            ctxBufPoolSetScrubSkip(pKernelChannelGroup->pChannelBufPool, NV_TRUE);
            ctxBufPoolSetScrubSkip(pKernelChannelGroup->pCtxBufPool, NV_TRUE);
            NV_PRINTF(LEVEL_INFO, "Skipping scrubber for all allocations on this context\n");
        }
    }

    //
    // If ctx buf pools are enabled, filter out partitionable engines
    // that aren't part of our instance.
    //
    // Memory needs to be reserved in the pool only for buffers for
    // engines in instance.
    //

    //
    // Size of memory that will be calculated for ctxBufPool reservation if ctxBufPool is enabled and MIG is disabled
    // or current engine belongs to this MIG instance and MIG is enabled
    //
    if (pKernelChannelGroup->pCtxBufPool != NULL &&
        (!bMIGInUse || kmigmgrIsEngineInInstance(pGpu, pKernelMIGManager, pKernelChannelGroup->engineType, ref)))
    {
        // GR Buffers
        if (RM_ENGINE_TYPE_IS_GR(pKernelChannelGroup->engineType))
        {
            KernelGraphics *pKernelGraphics = GPU_GET_KERNEL_GRAPHICS(pGpu, RM_ENGINE_TYPE_GR_IDX(pKernelChannelGroup->engineType));
            NvU32 bufId = 0;
            portMemSet(&bufInfoList[0], 0, sizeof(CTX_BUF_INFO) * NV_ENUM_SIZE(GR_CTX_BUFFER));
            bufCount = 0;

            kgraphicsDiscoverMaxLocalCtxBufferSize(pGpu, pKernelGraphics);

            FOR_EACH_IN_ENUM(GR_CTX_BUFFER, bufId)
            {
                // TODO expose engine class capabilities to kernel RM
                if (kgrmgrIsCtxBufSupported(bufId, NV_FALSE))
                {
                    const CTX_BUF_INFO *pBufInfo = kgraphicsGetCtxBufferInfo(pGpu, pKernelGraphics, bufId);
                    bufInfoList[bufCount] = *pBufInfo;
                    NV_PRINTF(LEVEL_INFO, "Reserving 0x%llx bytes for GR ctx bufId = %d\n",
                                  bufInfoList[bufCount].size, bufId);
                    bufCount++;
                }
            }
            FOR_EACH_IN_ENUM_END;
            bReserveMem = NV_TRUE;
        }
        else
        {
            // Allocate falcon context buffers if engine has (Kernel) Falcon object
            NvU32 ctxBufferSize;
            if (IS_GSP_CLIENT(pGpu))
            {
                ENGDESCRIPTOR engDesc;
                KernelFalcon *pKernelFalcon = NULL;

                NV_ASSERT_OK_OR_GOTO(rmStatus,
                    gpuXlateClientEngineIdToEngDesc(pGpu,
                                                    pKernelChannelGroup->engineType,
                                                    &engDesc),
                    failed);

                pKernelFalcon = kflcnGetKernelFalconForEngine(pGpu, engDesc);
                if (pKernelFalcon != NULL)
                {
                    ctxBufferSize = pKernelFalcon->ctxBufferSize;
                    bReserveMem = NV_TRUE;
                }
            }

            if (bReserveMem)
            {
                bufInfoList[0].size  = ctxBufferSize;
                bufInfoList[0].align = RM_PAGE_SIZE;
                bufInfoList[0].attr  = RM_ATTR_PAGE_SIZE_4KB;
                bufInfoList[0].bContig = NV_TRUE;
                NV_PRINTF(LEVEL_INFO, "Reserving 0x%llx bytes for engineType %d (%d) flcn ctx buffer\n",
                              bufInfoList[0].size, gpuGetNv2080EngineType(pKernelChannelGroup->engineType),
                              pKernelChannelGroup->engineType);
                bufCount++;
            }
            else
            {
                NV_PRINTF(LEVEL_INFO, "No buffer reserved for engineType %d (%d) in ctx_buf_pool\n",
                                  gpuGetNv2080EngineType(pKernelChannelGroup->engineType),
                                  pKernelChannelGroup->engineType);
            }
        }
    }

    if ((!bMIGInUse || RM_ENGINE_TYPE_IS_GR(pKernelChannelGroup->engineType))
        && !IsT234D(pGpu))
    {
        NV_ASSERT_OK_OR_GOTO(rmStatus,
            pRmApi->AllocWithSecInfo(pRmApi,
                pParams->hClient,
                RES_GET_HANDLE(pKernelChannelGroupApi),
                &pKernelChannelGroupApi->hKernelGraphicsContext,
                KERNEL_GRAPHICS_CONTEXT,
                NvP64_NULL,
                0,
                RMAPI_ALLOC_FLAGS_SKIP_RPC,
                NvP64_NULL,
                &pRmApi->defaultSecInfo),
            failed);
    }

    NV_PRINTF(LEVEL_INFO, "Adding group Id: %d hClient:0x%x\n",
              pKernelChannelGroup->grpID, pParams->hClient);

    if ((IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu)) &&
        !(pParams->allocFlags & RMAPI_ALLOC_FLAGS_SKIP_RPC))
    {
        NV_RM_RPC_ALLOC_OBJECT(pGpu,
                               pParams->hClient,
                               pParams->hParent,
                               pParams->hResource,
                               pParams->externalClassId,
                               pAllocParams,
                               sizeof(*pAllocParams),
                               rmStatus);
        //
        // Make sure that corresponding RPC occurs when freeing
        // KernelChannelGroupApi. Resource server checks this variable during
        // free and ignores any RPC flags set in resource_list.h
        //
        staticCast(pKernelChannelGroupApi, RmResource)->bRpcFree = NV_TRUE;

        if (rmStatus != NV_OK)
        {
            NV_PRINTF(LEVEL_ERROR,
                      "KernelChannelGroupApi alloc RPC to vGpu Host failed\n");
            goto failed;
        }

        if (IS_VIRTUAL_WITH_FULL_SRIOV(pGpu) || IS_GSP_CLIENT(pGpu))
        {
            NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS params = {
                0};
            NvU32 runqueueIdx;
            NvU32 maxRunqueues = kfifoGetNumRunqueues_HAL(pGpu, pKernelFifo);

            for (runqueueIdx = 0; runqueueIdx < maxRunqueues; ++runqueueIdx)
            {
                MEMORY_DESCRIPTOR          *pSrcMemDesc;
                HW_ENG_FAULT_METHOD_BUFFER *pMthdBuffer;
                pMthdBuffer = &pKernelChannelGroup->pMthdBuffers[runqueueIdx];
                pSrcMemDesc = pMthdBuffer->pMemDesc;

                params.methodBufferMemdesc[runqueueIdx].size = (
                    pSrcMemDesc->Size);
                params.methodBufferMemdesc[runqueueIdx].addressSpace = (
                    memdescGetAddressSpace(pSrcMemDesc));
                params.methodBufferMemdesc[runqueueIdx].cpuCacheAttrib = (
                    memdescGetCpuCacheAttrib(pSrcMemDesc));
                params.methodBufferMemdesc[runqueueIdx].alignment = 1;

                if (IS_VIRTUAL_WITH_FULL_SRIOV(pGpu))
                {
                    params.bar2Addr[runqueueIdx] = pMthdBuffer->bar2Addr;
                    params.methodBufferMemdesc[runqueueIdx].base = (
                        memdescGetPhysAddr(pSrcMemDesc, AT_CPU, 0));
                }
                else
                {
                    //
                    // The case of both vGpu full SRIOV + GSP_CLIENT host is not
                    // supported. This else branch considers the case of
                    // GSP_CLIENT only without vGpu.
                    //
                    params.methodBufferMemdesc[runqueueIdx].base = (
                        memdescGetPhysAddr(pSrcMemDesc, AT_GPU, 0));
                }
            }
            params.numValidEntries = runqueueIdx;

            rmStatus = pRmApi->Control(pRmApi,
                pParams->hClient,
                RES_GET_HANDLE(pKernelChannelGroupApi),
                NVA06C_CTRL_CMD_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS,
                &params,
                sizeof params);

            if (rmStatus != NV_OK)
            {
                NV_PRINTF(LEVEL_ERROR,
                    "Control call to update method buffer memdesc failed\n");
                goto failed;
            }
        }
    }

    if (kfifoIsZombieSubctxWarEnabled(pKernelFifo))
    {
        kchangrpSetSubcontextZombieState_HAL(pGpu, pKernelChannelGroup, 0, NV_TRUE);
        kchangrpUpdateSubcontextMask_HAL(pGpu, pKernelChannelGroup, 0, NV_TRUE);
    }

    // initialize apiObjList with original client's KernelChannelGroupApi object
    listInit(&pKernelChannelGroup->apiObjList, portMemAllocatorGetGlobalNonPaged());

    if (listAppendValue(&pKernelChannelGroup->apiObjList, &pKernelChannelGroupApi) == NULL)
    {
        rmStatus = NV_ERR_INSUFFICIENT_RESOURCES;
        listClear(&pKernelChannelGroup->apiObjList);
        goto failed;
    }

failed:
    if (rmStatus != NV_OK)
    {
        if (pKernelChannelGroupApi->hKernelGraphicsContext != NV01_NULL_OBJECT)
        {
            pRmApi->Free(pRmApi, pParams->hClient,
                         pKernelChannelGroupApi->hKernelGraphicsContext);
        }

        if (pKernelChannelGroup != NULL)
        {
            if (bTsgAllocated)
                kchangrpDestroy(pGpu, pKernelChannelGroup);

            if (pKernelChannelGroup->pCtxBufPool != NULL)
                ctxBufPoolDestroy(&pKernelChannelGroup->pCtxBufPool);

            if (pKernelChannelGroup->pChannelBufPool != NULL)
                ctxBufPoolDestroy(&pKernelChannelGroup->pChannelBufPool);

        }

        if (pShared)
            serverFreeShare(&g_resServ, pShared);
    }

done:

    if (bLockAcquired)
        rmDeviceGpuLocksRelease(pGpu, GPUS_LOCK_FLAGS_NONE, NULL);

    if ((rmStatus == NV_OK) && bReserveMem)
    {
        // GPU lock should not be held when reserving memory for ctxBufPool
        NV_CHECK_OK(rmStatus, LEVEL_ERROR,
            ctxBufPoolReserve(pGpu, pKernelChannelGroup->pCtxBufPool, bufInfoList, bufCount));
        if (rmStatus != NV_OK)
        {
            // Acquire the lock again for the cleanup path
            NV_ASSERT_OK_OR_RETURN(rmDeviceGpuLocksAcquire(pGpu, GPUS_LOCK_FLAGS_NONE, RM_LOCK_MODULES_FIFO));
            bLockAcquired = NV_TRUE;
            goto failed;
        }
    }

    portMemFree(bufInfoList);

    return rmStatus;
}

NV_STATUS
kchangrpapiControl_IMPL
(
    KernelChannelGroupApi          *pKernelChannelGroupApi,
    CALL_CONTEXT                   *pCallContext,
    RS_RES_CONTROL_PARAMS_INTERNAL *pParams
)
{
    RsResourceRef *pResourceRef = RES_GET_REF(pKernelChannelGroupApi);

    (void)pResourceRef;
    NV_PRINTF(LEVEL_INFO, "grpID 0x%x handle 0x%x cmd 0x%x\n",
              pKernelChannelGroupApi->pKernelChannelGroup->grpID,
              pResourceRef->hResource, pParams->pLegacyParams->cmd);

    return gpuresControl_IMPL(staticCast(pKernelChannelGroupApi, GpuResource),
                              pCallContext, pParams);
}

void
kchangrpapiDestruct_IMPL
(
    KernelChannelGroupApi *pKernelChannelGroupApi
)
{
    CALL_CONTEXT           *pCallContext;
    RS_RES_FREE_PARAMS_INTERNAL *pParams;
    RsResourceRef          *pResourceRef;
    RsClient               *pClient;
    KernelChannelGroup *pKernelChannelGroup =
        pKernelChannelGroupApi->pKernelChannelGroup;
    OBJGPU                 *pGpu = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    NV_STATUS               rmStatus = NV_OK;
    RS_ORDERED_ITERATOR     it;
    RsShared               *pShared = staticCast(pKernelChannelGroup, RsShared);
    RM_API                 *pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);

    resGetFreeParams(staticCast(pKernelChannelGroupApi, RsResource),
                     &pCallContext, &pParams);
    pResourceRef = pCallContext->pResourceRef;
    pClient = pCallContext->pClient;

    NV_PRINTF(LEVEL_INFO, "\n");

    // GHOST: this handle is going away whether or not the shared group
    // survives it, and the scheduler's slot names the group by this handle.
    ghostSchedForget_GHOST(pClient->hClient, pResourceRef->hResource);

    // RS-TODO should still free channels?
    if (serverGetShareRefCount(&g_resServ, pShared) > 1)
    {
        // Remove this kchangrpapi object from the list of owners in the shared object
        listRemoveFirstByValue(&pKernelChannelGroupApi->pKernelChannelGroup->apiObjList, &pKernelChannelGroupApi);
        goto done;
    }

    if (pKernelChannelGroup != NULL)
        kchangrpSetRealtime_HAL(pGpu, pKernelChannelGroup, NV_FALSE);

    // If channels still exist in this group, free them
    // RS-TODO this can be removed after re-parenting support is added
    it = kchannelGetIter(pClient, pResourceRef);
    while (clientRefOrderedIterNext(pClient, &it))
    {
        NV_STATUS tmpStatus;

        tmpStatus = pRmApi->Free(pRmApi, pClient->hClient, it.pResourceRef->hResource);
        if ((tmpStatus != NV_OK) && (rmStatus == NV_OK))
            rmStatus = tmpStatus;
    }

    NV_ASSERT(rmStatus == NV_OK);

    if (pKernelChannelGroup != NULL)
    {
        kchangrpDestroy(pGpu, pKernelChannelGroup);

        if (pKernelChannelGroup->pCtxBufPool != NULL)
        {
            ctxBufPoolRelease(pKernelChannelGroup->pCtxBufPool);
            ctxBufPoolDestroy(&pKernelChannelGroup->pCtxBufPool);
        }

        if (pKernelChannelGroup->pChannelBufPool != NULL)
        {
            ctxBufPoolRelease(pKernelChannelGroup->pChannelBufPool);
            ctxBufPoolDestroy(&pKernelChannelGroup->pChannelBufPool);
        }

        listClear(&pKernelChannelGroup->apiObjList);
    }

done:
    serverFreeShare(&g_resServ, pShared);

    pParams->status = rmStatus;
}

NV_STATUS
kchangrpapiCopyConstruct_IMPL
(
    KernelChannelGroupApi        *pKernelChannelGroupApi,
    CALL_CONTEXT                 *pCallContext,
    RS_RES_ALLOC_PARAMS_INTERNAL *pParams
)
{
    RM_API *pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);
    RsClient *pDstClient = pCallContext->pClient;
    RsResourceRef *pDstRef = pCallContext->pResourceRef;
    RsResourceRef *pSrcRef = pParams->pSrcRef;
    KernelChannelGroupApi *pChanGrpSrc = dynamicCast(pSrcRef->pResource,
                                                     KernelChannelGroupApi);
    RS_ITERATOR iter;
    OBJGPU       *pGpu   = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    NV_STATUS     status = NV_OK;
    RsResourceRef *pVaspaceRef = NULL;
    VaSpaceApi *pVaspaceApi = NULL;

    pKernelChannelGroupApi->hKernelGraphicsContext  = NV01_NULL_OBJECT;
    pKernelChannelGroupApi->hLegacykCtxShareSync    = NV01_NULL_OBJECT;
    pKernelChannelGroupApi->hLegacykCtxShareAsync   = NV01_NULL_OBJECT;

    pKernelChannelGroupApi->pKernelChannelGroup =
        pChanGrpSrc->pKernelChannelGroup;
    serverRefShare(&g_resServ,
        staticCast(pKernelChannelGroupApi->pKernelChannelGroup, RsShared));

    iter =  serverutilRefIter(pDstClient->hClient, pDstRef->pParentRef->hResource, classId(VaSpaceApi), RS_ITERATE_DESCENDANTS, NV_TRUE);
    while (clientRefIterNext(iter.pClient, &iter))
    {
        pVaspaceRef = iter.pResourceRef;
        pVaspaceApi = dynamicCast(pVaspaceRef->pResource, VaSpaceApi);
        NV_ASSERT_OR_RETURN(pVaspaceApi != NULL, NV_ERR_INVALID_STATE);

        if (pVaspaceApi->pVASpace ==
            pKernelChannelGroupApi->pKernelChannelGroup->pVAS)
        {
            refAddDependant(pVaspaceRef, pDstRef);
            break;
        }
    }

    if (pChanGrpSrc->hKernelGraphicsContext != NV01_NULL_OBJECT)
    {
        NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
            pRmApi->DupObject(pRmApi,
                              pDstClient->hClient,
                              pDstRef->hResource,
                              &pKernelChannelGroupApi->hKernelGraphicsContext,
                              pParams->pSrcClient->hClient,
                              pChanGrpSrc->hKernelGraphicsContext,
                              0),
            fail);
    }

    //
    // If this channel group is in legacy mode, new client needs its own handles to the
    // sync and async internally allocated kctxshares
    //
    if (pChanGrpSrc->pKernelChannelGroup->bLegacyMode)
    {
        NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
            pRmApi->DupObject(pRmApi,
                              pDstClient->hClient,
                              pDstRef->hResource,
                              &pKernelChannelGroupApi->hLegacykCtxShareSync,
                              pParams->pSrcClient->hClient,
                              pChanGrpSrc->hLegacykCtxShareSync,
                              0),
            fail);

        // All chips have SYNC, Some chips won't have an ASYNC kctxshare
        if (pChanGrpSrc->hLegacykCtxShareAsync != 0)
        {
            NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
                pRmApi->DupObject(pRmApi,
                                  pDstClient->hClient,
                                  pDstRef->hResource,
                                  &pKernelChannelGroupApi->hLegacykCtxShareAsync,
                                  pParams->pSrcClient->hClient,
                                  pChanGrpSrc->hLegacykCtxShareAsync,
                                  0),
            fail);
        }
    }

    if (IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu))
    {
        NV_RM_RPC_DUP_OBJECT(pGpu, pDstClient->hClient, pDstRef->pParentRef->hResource, pDstRef->hResource,
                             pParams->pSrcClient->hClient, pSrcRef->hResource, 0,
                             NV_TRUE, // automatically issue RPC_FREE on object free
                             pDstRef, status);

        if (status != NV_OK)
            goto fail;
    }

    if (listAppendValue(&pKernelChannelGroupApi->pKernelChannelGroup->apiObjList, &pKernelChannelGroupApi) == NULL)
    {
        status = NV_ERR_INSUFFICIENT_RESOURCES;
        goto fail;
    }

    return status;

fail:
    if (pKernelChannelGroupApi->hLegacykCtxShareAsync != NV01_NULL_OBJECT)
    {
        pRmApi->Free(pRmApi, pDstClient->hClient,
                     pKernelChannelGroupApi->hLegacykCtxShareAsync);
    }
    if (pKernelChannelGroupApi->hLegacykCtxShareSync != NV01_NULL_OBJECT)
    {
        pRmApi->Free(pRmApi, pDstClient->hClient,
                     pKernelChannelGroupApi->hLegacykCtxShareSync);
    }
    if (pKernelChannelGroupApi->hKernelGraphicsContext != NV01_NULL_OBJECT)
    {
        pRmApi->Free(pRmApi, pDstClient->hClient,
                     pKernelChannelGroupApi->hKernelGraphicsContext);
    }

    serverFreeShare(&g_resServ,
        staticCast(pKernelChannelGroupApi->pKernelChannelGroup, RsShared));

    return status;
}

NvBool
kchangrpapiCanCopy_IMPL
(
    KernelChannelGroupApi *pKernelChannelGroupApi
)
{
    return NV_TRUE;
}

NV_STATUS
CliGetChannelGroup
(
    NvHandle                 hClient,
    NvHandle                 hChanGrp,
    RsResourceRef          **ppChanGrpRef,
    NvHandle                *phDevice
)
{
    NV_STATUS status;
    RsClient *pRsClient;
    RsResourceRef *pResourceRef;
    RsResourceRef *pParentRef;

    if (!ppChanGrpRef)
    {
        return NV_ERR_INVALID_ARGUMENT;
    }

    status = serverGetClientUnderLock(&g_resServ, hClient, &pRsClient);
    NV_ASSERT(status == NV_OK);
    if (status != NV_OK)
        return status;

    status = clientGetResourceRefByType(pRsClient, hChanGrp,
                                        classId(KernelChannelGroupApi),
                                        &pResourceRef);
    if (status != NV_OK)
        return status;

    *ppChanGrpRef = pResourceRef;

    if (phDevice)
    {
        pParentRef = pResourceRef->pParentRef;
        *phDevice = pParentRef->hResource;
    }

    return NV_OK;
}

/*!
 * @brief Use TSG in legacy mode
 *
 * In legacy mode, RM pre-allocates the subcontexts in a TSG.
 * This is needed for the following reasons:
 *
 *  1. We are also using subcontext to represent TSG contexts in pre-VOLTA chips (see below).
 *     But RM clients haven't yet moved to the subcontext model in production code.
 *     So RM implicitly creates it for them, until they make the switch.
 *
 *  2. Pre-VOLTA, we only support one address space in a TSG.
 *     Preallocating the subcontext prevents accidental use of multiple address spaces within a TSG.
 *     So we use the vaspace specified/implied at TSG creation to create the subcontexts.
 *
 *  3. Tests and clients on VOLTA that don't explicitly specify subcontexts need to behave similar
 *     to previous chips until they allocate the kctxshares themselves.
 *
 *  Legacy subcontexts are interpreted in the following ways:
 *
 *     VOLTA+            : subcontext 0 is VEID 0, subcontext 1 is VEID 1
 *     GM20X thru PASCAL : subcontext 0 is SCG type 0, subcontext 1 is SCG type 1
 *     pre-GM20X         : just a single subcontext 0; no SCG or VEIDs attached to it.
 *
 * @param[in] pKernelChannelGroupApi Channel group pointer
 * @param[in] pGpu                   GPU object pointer
 * @param[in] pKernelFifo            FIFO object pointer
 * @param[in] hClient                Client handle
 *
 */
NV_STATUS
kchangrpapiSetLegacyMode_IMPL
(
    KernelChannelGroupApi *pKernelChannelGroupApi,
    OBJGPU                *pGpu,
    KernelFifo            *pKernelFifo,
    NvHandle               hClient
)
{
    KernelChannelGroup *pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;
    NvHandle hTsg = RES_GET_HANDLE(pKernelChannelGroupApi);
    NvHandle hkCtxShare = 0;
    NV_STATUS status = NV_OK;
    NvU32 maxSubctx = 0;
    NvU64 numMax = 0;
    NvU64 numFree = 0;
    RM_API *pRmApi = rmapiGetInterface(RMAPI_GPU_LOCK_INTERNAL);
    KernelChannelGroupApiListIter it;

    NV_CTXSHARE_ALLOCATION_PARAMETERS kctxshareParams = { 0 };

    ct_assert(NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC == 0);
    ct_assert(NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC == 1);

    NV_ASSERT_OK(pKernelChannelGroup->pSubctxIdHeap->eheapGetSize(
        pKernelChannelGroup->pSubctxIdHeap,
        &numMax));

    NV_ASSERT_OK(pKernelChannelGroup->pSubctxIdHeap->eheapGetFree(
        pKernelChannelGroup->pSubctxIdHeap,
        &numFree));

    NV_ASSERT(numMax ==
              kfifoChannelGroupGetLocalMaxSubcontext_HAL(pGpu, pKernelFifo,
                                                         pKernelChannelGroup,
                                                         NV_FALSE));

    NV_ASSERT_OR_RETURN(numMax == numFree && numMax != 0, NV_ERR_INVALID_STATE);

    pKernelChannelGroup->pSubctxIdHeap->eheapDestruct(
        pKernelChannelGroup->pSubctxIdHeap);
    pKernelChannelGroup->pVaSpaceIdHeap->eheapDestruct(
        pKernelChannelGroup->pVaSpaceIdHeap);
    //
    // There should only be 1 (SYNC) or 2 legacy kctxshares (SYNC + ASYNC),
    // depending on chip
    //
    maxSubctx = kfifoChannelGroupGetLocalMaxSubcontext_HAL(pGpu, pKernelFifo,
                                                           pKernelChannelGroup,
                                                           NV_TRUE);
    NV_ASSERT_OR_RETURN(numMax == numFree, NV_ERR_INVALID_STATE);
    NV_ASSERT(maxSubctx == 1 || maxSubctx == 2);

    constructObjEHeap(pKernelChannelGroup->pSubctxIdHeap,
                      0, maxSubctx, sizeof(KernelCtxShare *), 0);
    constructObjEHeap(pKernelChannelGroup->pVaSpaceIdHeap,
                      0, maxSubctx, sizeof(KernelCtxShare *), 0);

    pKernelChannelGroup->bLegacyMode = NV_TRUE;

    // Allocate SYNC
    hkCtxShare = 0;
    kctxshareParams.hVASpace = 0;
    kctxshareParams.flags    = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC;
    kctxshareParams.subctxId = 0xFFFFFFFF;

    NV_ASSERT_OK_OR_GOTO(status,
                         pRmApi->AllocWithSecInfo(pRmApi,
                                                  hClient,
                                                  hTsg,
                                                  &hkCtxShare,
                                                  FERMI_CONTEXT_SHARE_A,
                                                  NV_PTR_TO_NvP64(&kctxshareParams),
                                                  sizeof(kctxshareParams),
                                                  RMAPI_ALLOC_FLAGS_SKIP_RPC,
                                                  NvP64_NULL,
                                                  &pRmApi->defaultSecInfo),
                         fail);

    NV_ASSERT(kctxshareParams.subctxId == NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC);

    pKernelChannelGroupApi->hLegacykCtxShareSync = hkCtxShare;

    if(maxSubctx == 2)
    {
        // Allocate ASYNC
        hkCtxShare = 0;
        kctxshareParams.hVASpace = 0;
        kctxshareParams.flags    = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
        kctxshareParams.subctxId = 0xFFFFFFFF;

        NV_ASSERT_OK_OR_GOTO(status,
                             pRmApi->AllocWithSecInfo(pRmApi,
                                                      hClient,
                                                      hTsg,
                                                      &hkCtxShare,
                                                      FERMI_CONTEXT_SHARE_A,
                                                      NV_PTR_TO_NvP64(&kctxshareParams),
                                                      sizeof(kctxshareParams),
                                                      RMAPI_ALLOC_FLAGS_SKIP_RPC,
                                                      NvP64_NULL,
                                                      &pRmApi->defaultSecInfo),
                             fail);

        NV_ASSERT(kctxshareParams.subctxId == NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC);

        pKernelChannelGroupApi->hLegacykCtxShareAsync = hkCtxShare;
    }

    NV_ASSERT_OK_OR_GOTO(status,
                         pKernelChannelGroup->pSubctxIdHeap->eheapGetFree(
                             pKernelChannelGroup->pSubctxIdHeap,
                             &numFree),
                         fail);

    NV_ASSERT_OR_GOTO(numFree == 0, fail);

    //
    // If this channel group has been duped, we need to provide kctxshareApi handles to the
    // other channelGroupApi objects that share this channel group since the handles will
    // only work for a single client.
    //
    it = listIterAll(&pKernelChannelGroup->apiObjList);
    while (listIterNext(&it))
    {
        KernelChannelGroupApi *pChanGrpDest = *it.pValue;

        if(pChanGrpDest == pKernelChannelGroupApi)
            continue;

        NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
            pRmApi->DupObject(pRmApi,
                              RES_GET_CLIENT_HANDLE(pChanGrpDest),
                              RES_GET_HANDLE(pChanGrpDest),
                              &pChanGrpDest->hLegacykCtxShareSync,
                              RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi),
                              pKernelChannelGroupApi->hLegacykCtxShareSync,
                              0),
            fail);

        if (maxSubctx == 2)
        {
            NV_CHECK_OK_OR_GOTO(status, LEVEL_ERROR,
                pRmApi->DupObject(pRmApi,
                                  RES_GET_CLIENT_HANDLE(pChanGrpDest),
                                  RES_GET_HANDLE(pChanGrpDest),
                                  &pChanGrpDest->hLegacykCtxShareAsync,
                                  RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi),
                                  pKernelChannelGroupApi->hLegacykCtxShareAsync,
                                  0),
            fail);
        }
    }

    return status;

fail:
    NV_PRINTF(LEVEL_ERROR, "Failed to set channel group in legacy mode.\n");

    pKernelChannelGroup->bLegacyMode = NV_FALSE;

    it = listIterAll(&pKernelChannelGroup->apiObjList);

    while (listIterNext(&it))
    {
        KernelChannelGroupApi *pChanGrpIt = *it.pValue;

        if (pChanGrpIt->hLegacykCtxShareSync != 0)
        {
           pRmApi->Free(pRmApi, RES_GET_CLIENT_HANDLE(pChanGrpIt), pChanGrpIt->hLegacykCtxShareSync);
           pChanGrpIt->hLegacykCtxShareSync = 0;
        }

        if (pChanGrpIt->hLegacykCtxShareAsync != 0)
        {
           pRmApi->Free(pRmApi, RES_GET_CLIENT_HANDLE(pChanGrpIt), pChanGrpIt->hLegacykCtxShareAsync);
           pChanGrpIt->hLegacykCtxShareAsync = 0;
        }
    }

    if(status == NV_OK)
    {
        status = NV_ERR_INVALID_STATE;
    }

    return status;
}

NV_STATUS
kchangrpapiCtrlCmdGpFifoSchedule_IMPL
(
    KernelChannelGroupApi              *pKernelChannelGroupApi,
    NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS *pSchedParams
)
{
    OBJGPU              *pGpu         = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    RsResourceRef       *pResourceRef = RES_GET_REF(pKernelChannelGroupApi);
    KernelChannelGroup  *pKernelChannelGroup = NULL;
    NV_STATUS            status       = NV_OK;
    KernelFifo          *pKernelFifo;
    CLASSDESCRIPTOR     *pClass       = NULL;
    CHANNEL_NODE        *pChanNode    = NULL;
    CHANNEL_LIST        *pChanList    = NULL;
    NvU32                runlistId    = INVALID_RUNLIST_ID;
    RM_API              *pRmApi       = GPU_GET_PHYSICAL_RMAPI(pGpu);

    if (pKernelChannelGroupApi->pKernelChannelGroup == NULL)
        return NV_ERR_INVALID_OBJECT;
    pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;

    if (gpuGetClassByClassId(pGpu, pResourceRef->externalClassId, &pClass) != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "class %x not supported\n",
                  pResourceRef->externalClassId);
    }
    NV_ASSERT_OR_RETURN((pClass != NULL), NV_ERR_NOT_SUPPORTED);

    //
    // Bug 1737765: Prevent Externally Owned Channels from running unless bound
    //  It is possible for clients to allocate and schedule channels while
    //  skipping the UVM registration step which binds the appropriate
    //  allocations in RM. We need to fail channel scheduling if the channels
    //  have not been registered with UVM.
    //  We include this check for every channel in the group because it is
    //  expected that Volta+ may use a separate VAS for each channel.
    //

    pChanList = pKernelChannelGroup->pChanList;

    for (pChanNode = pChanList->pHead; pChanNode; pChanNode = pChanNode->pNext)
    {
        NV_CHECK_OR_RETURN(LEVEL_NOTICE, kchannelIsSchedulable_HAL(pGpu, pChanNode->pKernelChannel),
            NV_ERR_INVALID_STATE);
    }

    SLI_LOOP_START(SLI_LOOP_FLAGS_BC_ONLY);
    pKernelFifo = GPU_GET_KERNEL_FIFO(pGpu);
    pChanList = pKernelChannelGroup->pChanList;

    //
    // Some channels may not have objects allocated on them, so they won't have
    // a runlist committed yet.  Force them all onto the same runlist so the
    // low level code knows what do to with them.
    //
    // First we walk through the channels to see if there is a runlist assigned
    // already and if so are the channels consistent.
    //
    runlistId = pKernelChannelGroup->runlistId; // Start with TSG runlistId
    for (pChanNode = pChanList->pHead; pChanNode; pChanNode = pChanNode->pNext)
    {
        KernelChannel *pKernelChannel = pChanNode->pKernelChannel;

        NV_ASSERT_OR_ELSE(pKernelChannel != NULL, continue);

        if (kchannelIsRunlistSet(pGpu, pKernelChannel))
        {
            if (runlistId == INVALID_RUNLIST_ID)
            {
                runlistId = kchannelGetRunlistId(pKernelChannel);
            }
            else // Catch if 2 channels in the same TSG have different runlistId
            {
                if (runlistId != kchannelGetRunlistId(pKernelChannel))
                {
                    NV_PRINTF(LEVEL_ERROR,
                        "Channels in TSG %d have different runlist IDs this should never happen!\n",
                        pKernelChannelGroup->grpID);
                    DBG_BREAKPOINT();
                }
            }
        }
    }

    // If no channels have a runlist set, get the default and use it.
    if (runlistId == INVALID_RUNLIST_ID)
    {
        runlistId = kfifoGetDefaultRunlist_HAL(pGpu, pKernelFifo,
            pKernelChannelGroup->engineType);
    }

    // We can rewrite TSG runlist id just as we will do that for all TSG channels below
    pKernelChannelGroup->runlistId = runlistId;

    //
    // Now go through and force any channels w/o the runlist set to use either
    // the default or whatever we found other channels to be allocated on.
    //
    for (pChanNode = pChanList->pHead; pChanNode; pChanNode = pChanNode->pNext)
    {
        KernelChannel *pKernelChannel = pChanNode->pKernelChannel;

        NV_ASSERT_OR_ELSE(pKernelChannel != NULL, continue);

        if (!kchannelIsRunlistSet(pGpu, pKernelChannel))
        {
            kfifoRunlistSetId_HAL(pGpu, pKernelFifo, pKernelChannel, runlistId);
        }
    }
    SLI_LOOP_END

    if (IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu))
    {
        CALL_CONTEXT *pCallContext = resservGetTlsCallContext();
        RmCtrlParams *pRmCtrlParams = pCallContext->pControlParams;
        NvHandle hClient = RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi);
        NvHandle hObject = RES_GET_HANDLE(pKernelChannelGroupApi);

        NV_RM_RPC_CONTROL(pGpu,
                          hClient,
                          hObject,
                          pRmCtrlParams->cmd,
                          pRmCtrlParams->pParams,
                          pRmCtrlParams->paramsSize,
                          status);

        // === GHOST EXPERIMENT (Phase 0b): after the sandbox enables its channel
        // group, force it back OFF from inside RM, to test whether a driver-level
        // TSG detach STICKS against a doorbell workload on Blackwell's GSP -- the
        // userspace GPFIFO_SCHEDULE(disable) did not. If the burn pod then makes
        // no progress (no matmuls, GPU idle), a driver detach sticks and
        // Ghost-style time-division is viable here; if it runs at full rate, the
        // detach is overridden, same as userspace. Compile-time toggle.
        {
            // Runtime knob: insmod nvidia.ko NVreg_RegistryDwords="GhostDetach=1"
            NvU32  ghostDetachReg = 0;
            NvBool bGhostDetach;
            if (osReadRegistryDword(pGpu, "GhostDetach", &ghostDetachReg) != NV_OK)
                ghostDetachReg = 0;
            bGhostDetach = (ghostDetachReg != 0);
            if (bGhostDetach && (pSchedParams != NULL) && pSchedParams->bEnable && (status == NV_OK))
            {
                NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS off;
                NV_STATUS ds;
                portMemSet(&off, 0, sizeof(off));
                off.bEnable = NV_FALSE;
                ds = pRmApi->Control(pRmApi, hClient, hObject,
                                     NVA06C_CTRL_CMD_INTERNAL_GPFIFO_SCHEDULE,
                                     &off, sizeof(off));
                NV_PRINTF(LEVEL_ERROR,
                          "GHOST 0b: force-disabled TSG client 0x%08x obj 0x%08x after enable -> 0x%x\n",
                          hClient, hObject, ds);
            }
        }

        // === GHOST EXPERIMENT (0g): per-TSG timeslice, originated at kernel
        // privilege, weighted per tenant. On GB205 a 16:1 timeslice ratio set
        // from userspace divided the GPU 1:1 (GSP ignored it). This asks the
        // same question of a datacenter GSP: does the runlist actually honour
        // per-TSG timeslice? Knobs:
        //   GhostTimeslice  = microseconds for tenant 0 (0 => do not touch)
        //   GhostTimesliceB = microseconds for every later tenant
        {
            NvU32 tsA = 0, tsB = 0;
            if (osReadRegistryDword(pGpu, "GhostTimeslice", &tsA) != NV_OK)
                tsA = 0;
            if (osReadRegistryDword(pGpu, "GhostTimesliceB", &tsB) != NV_OK)
                tsB = 0;
            if ((tsA != 0) && (pSchedParams != NULL) && pSchedParams->bEnable &&
                (status == NV_OK))
            {
                NVA06C_CTRL_TIMESLICE_PARAMS ts;
                NV_STATUS sSet, sGet;
                NvU32 tenant = ghostTenantIndex_GHOST();
                NvU64 want = ((tenant == 0) || (tsB == 0)) ? tsA : tsB;

                portMemSet(&ts, 0, sizeof(ts));
                ts.timesliceUs = want;
                sSet = pRmApi->Control(pRmApi, hClient, hObject,
                                       NVA06C_CTRL_CMD_SET_TIMESLICE,
                                       &ts, sizeof(ts));
                portMemSet(&ts, 0, sizeof(ts));
                sGet = pRmApi->Control(pRmApi, hClient, hObject,
                                       NVA06C_CTRL_CMD_GET_TIMESLICE,
                                       &ts, sizeof(ts));
                NV_PRINTF(LEVEL_ERROR,
                          "GHOST 0g: tenant %u TSG 0x%08x SET_TIMESLICE(%llu us)=0x%x "
                          "GET=0x%x readback=%llu us\n",
                          tenant, hObject, want, sSet, sGet, ts.timesliceUs);
            }
        }

        // === GHOST EXPERIMENT (0h): runlist interleave level, the one temporal
        // lever never tested on GB205 (it was admin-gated from the Sentry, and
        // we could not set it). At kernel privilege we can. A HIGH TSG is
        // supposed to appear (M+1)*L times in the runlist against a LOW TSG's
        // one, so with the tenants' TSG counts equal this should skew service.
        //   GhostInterleave  = level for tenant 0 (0 LOW, 1 MEDIUM, 2 HIGH;
        //                      255 => do not touch)
        //   GhostInterleaveB = level for every later tenant
        {
            NvU32 ilA = 0xff, ilB = 0xff;
            if (osReadRegistryDword(pGpu, "GhostInterleave", &ilA) != NV_OK)
                ilA = 0xff;
            if (osReadRegistryDword(pGpu, "GhostInterleaveB", &ilB) != NV_OK)
                ilB = 0xff;
            if ((ilA != 0xff) && (pSchedParams != NULL) && pSchedParams->bEnable &&
                (status == NV_OK))
            {
                NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS il;
                NV_STATUS sSet, sGet;
                NvU32 tenant = ghostTenantIndex_GHOST();
                NvU32 want = ((tenant == 0) || (ilB == 0xff)) ? ilA : ilB;

                portMemSet(&il, 0, sizeof(il));
                il.tsgInterleaveLevel = want;
                sSet = pRmApi->Control(pRmApi, hClient, hObject,
                                       NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
                                       &il, sizeof(il));
                portMemSet(&il, 0, sizeof(il));
                sGet = pRmApi->Control(pRmApi, hClient, hObject,
                                       NVA06C_CTRL_CMD_GET_INTERLEAVE_LEVEL,
                                       &il, sizeof(il));
                NV_PRINTF(LEVEL_ERROR,
                          "GHOST 0h: tenant %u TSG 0x%08x SET_INTERLEAVE(%u)=0x%x "
                          "GET=0x%x readback=%u\n",
                          tenant, hObject, want, sSet, sGet, il.tsgInterleaveLevel);
            }
        }

        // === GHOST GPU SCHEDULER: record this tenant's channels so a later
        // detach/attach (driven by the userspace scheduler via
        // /proc/driver/nvidia/gpusched) can find them. The detach itself is no
        // longer done here unconditionally; it is issued on command. The
        // GhostPreempt knob is kept below as a static self-test only.
        ghostSchedRecord_GHOST(pGpu, osGetCurrentProcess(), hClient, hObject,
                               pKernelChannelGroupApi->pKernelChannelGroup);
        {
            NvU32 ghostPreempt = 0;
            if (osReadRegistryDword(pGpu, "GhostPreempt", &ghostPreempt) != NV_OK)
                ghostPreempt = 0;
            if ((ghostPreempt != 0) && (pSchedParams != NULL) &&
                pSchedParams->bEnable && (status == NV_OK) &&
                (ghostTenantIndex_GHOST() >= 1))
            {
                ghostSchedQueue_GHOST(osGetCurrentProcess(), 1, 0); // 1=detach
            }
        }


        // === GHOST EXPERIMENT (0e): re-issue the 0c/0d controls now that the
        // ctxshares are fully constructed and registered with GSP. Inside their
        // own constructor they returned OBJECT_NOT_FOUND (0x57), which says
        // nothing about feature support; this call site is the control.
        if ((pSchedParams != NULL) && pSchedParams->bEnable && (status == NV_OK))
        {
            ghostReprobeDeferred_GHOST(pGpu);
        }
        return status;
    }


    //
    // Do an internal control call to do channel reset
    // on Host (Physical) RM
    //
    status = pRmApi->Control(pRmApi,
                             RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi),
                             RES_GET_HANDLE(pKernelChannelGroupApi),
                             NVA06C_CTRL_CMD_INTERNAL_GPFIFO_SCHEDULE,
                             pSchedParams,
                             sizeof(NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS));

    return status;
}

NV_STATUS
kchangrpapiCtrlCmdBind_IMPL
(
    KernelChannelGroupApi   *pKernelChannelGroupApi,
    NVA06C_CTRL_BIND_PARAMS *pParams
)
{
    NV_STATUS     rmStatus = NV_OK;
    OBJGPU       *pGpu     = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    Device       *pDevice  = GPU_RES_GET_DEVICE(pKernelChannelGroupApi);
    CHANNEL_NODE *pChanNode;
    RM_ENGINE_TYPE localEngineType;
    RM_ENGINE_TYPE globalEngineType;
    ENGDESCRIPTOR engineDesc;
    NvBool        bMIGInUse = IS_MIG_IN_USE(pGpu);

    NV_ASSERT_OR_RETURN(pParams != NULL, NV_ERR_INVALID_ARGUMENT);

    localEngineType = globalEngineType = gpuGetRmEngineType(pParams->engineType);

    if (bMIGInUse)
    {
        KernelMIGManager *pKernelMIGManager = GPU_GET_KERNEL_MIG_MANAGER(pGpu);
        MIG_INSTANCE_REF ref;

        NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
            kmigmgrGetInstanceRefFromDevice(pGpu, pKernelMIGManager, pDevice, &ref));

        NV_CHECK_OK_OR_RETURN(LEVEL_ERROR,
            kmigmgrGetLocalToGlobalEngineType(pGpu, pKernelMIGManager, ref,
                                              localEngineType,
                                              &globalEngineType));
    }

    NV_PRINTF(LEVEL_INFO,
              "Binding TSG %d to Engine %d (%d)\n",
              pKernelChannelGroupApi->pKernelChannelGroup->grpID,
              gpuGetNv2080EngineType(globalEngineType), globalEngineType);

    // Translate globalEnginetype -> enginedesc
    NV_ASSERT_OK_OR_CAPTURE_FIRST_ERROR(rmStatus,
        gpuXlateClientEngineIdToEngDesc(pGpu, globalEngineType, &engineDesc));

    // Translate engineDesc -> runlistId for TSG
    NV_ASSERT_OK_OR_CAPTURE_FIRST_ERROR(rmStatus,
        kfifoEngineInfoXlate_HAL(pGpu, GPU_GET_KERNEL_FIFO(pGpu),
            ENGINE_INFO_TYPE_ENG_DESC,
            engineDesc,
            ENGINE_INFO_TYPE_RUNLIST,
            &pKernelChannelGroupApi->pKernelChannelGroup->runlistId));

    for (pChanNode =
             pKernelChannelGroupApi->pKernelChannelGroup->pChanList->pHead;
         pChanNode != NULL;
         pChanNode = pChanNode->pNext)
    {
        NV_ASSERT_OK_OR_CAPTURE_FIRST_ERROR(rmStatus,
            kchannelBindToRunlist(pChanNode->pKernelChannel,
                                  localEngineType,
                                  engineDesc));
        if (rmStatus != NV_OK)
        {
            break;
        }
    }

    return rmStatus;
}

NV_STATUS
kchangrpapiCtrlCmdGetTimeslice_IMPL
(
    KernelChannelGroupApi        *pKernelChannelGroupApi,
    NVA06C_CTRL_TIMESLICE_PARAMS *pTsParams
)
{
    KernelChannelGroup *pKernelChannelGroup = NULL;

    if (pKernelChannelGroupApi->pKernelChannelGroup == NULL)
        return NV_ERR_INVALID_OBJECT;
    pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;

    pTsParams->timesliceUs = pKernelChannelGroup->timesliceUs;

    return NV_OK;
}

NV_STATUS
kchangrpapiCtrlCmdSetTimeslice_IMPL
(
    KernelChannelGroupApi        *pKernelChannelGroupApi,
    NVA06C_CTRL_TIMESLICE_PARAMS *pTsParams
)
{
    OBJGPU             *pGpu                = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    RsResourceRef      *pResourceRef        = RES_GET_REF(pKernelChannelGroupApi);
    KernelChannelGroup *pKernelChannelGroup = NULL;
    NV_STATUS           status              = NV_OK;
    CLASSDESCRIPTOR    *pClass              = NULL;
    RM_API             *pRmApi              = GPU_GET_PHYSICAL_RMAPI(pGpu);

    if (pKernelChannelGroupApi->pKernelChannelGroup == NULL)
        return NV_ERR_INVALID_OBJECT;
    pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;

    if (gpuGetClassByClassId(pGpu, pResourceRef->externalClassId, &pClass) != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "class %x not supported\n",
                  pResourceRef->externalClassId);
    }
    NV_ASSERT_OR_RETURN((pClass != NULL), NV_ERR_NOT_SUPPORTED);

    if (IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu))
    {
        CALL_CONTEXT *pCallContext = resservGetTlsCallContext();
        RmCtrlParams *pRmCtrlParams = pCallContext->pControlParams;
        NvHandle hClient = RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi);
        NvHandle hObject = RES_GET_HANDLE(pKernelChannelGroupApi);
        NVA06C_CTRL_TIMESLICE_PARAMS *pParams = (NVA06C_CTRL_TIMESLICE_PARAMS *)(pRmCtrlParams->pParams);

        NV_RM_RPC_CONTROL(pGpu,
                          hClient,
                          hObject,
                          pRmCtrlParams->cmd,
                          pRmCtrlParams->pParams,
                          pRmCtrlParams->paramsSize,
                          status);

        // Update guest RM's internal bookkeeping with the timeslice.
        if (status == NV_OK)
        {
            pKernelChannelGroup->timesliceUs = pParams->timesliceUs;
        }

        return status;
    }

    //
    // Do an internal control call to do channel reset
    // on Host (Physical) RM
    //
    status = pRmApi->Control(pRmApi,
                             RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi),
                             RES_GET_HANDLE(pKernelChannelGroupApi),
                             NVA06C_CTRL_CMD_INTERNAL_SET_TIMESLICE,
                             pTsParams,
                             sizeof(NVA06C_CTRL_TIMESLICE_PARAMS));

    return status;
}

NV_STATUS
kchangrpapiCtrlCmdGetInfo_IMPL
(
    KernelChannelGroupApi       *pKernelChannelGroupApi,
    NVA06C_CTRL_GET_INFO_PARAMS *pParams
)
{
    KernelChannelGroup *pKernelChannelGroup = NULL;

    if (pKernelChannelGroupApi->pKernelChannelGroup == NULL)
        return NV_ERR_INVALID_OBJECT;
    pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;

    pParams->tsgID = pKernelChannelGroup->grpID;

    return NV_OK;
}

NV_STATUS
kchangrpapiCtrlCmdSetInterleaveLevel_IMPL
(
    KernelChannelGroupApi               *pKernelChannelGroupApi,
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS *pParams
)
{
    OBJGPU          *pGpu         = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    RsResourceRef   *pResourceRef = RES_GET_REF(pKernelChannelGroupApi);
    KernelChannelGroup *pKernelChannelGroup =
        pKernelChannelGroupApi->pKernelChannelGroup;
    CLASSDESCRIPTOR *pClass       = NULL;
    NV_STATUS        status       = NV_OK;

    if (gpuGetClassByClassId(pGpu, pResourceRef->externalClassId, &pClass) != NV_OK)
    {
        NV_PRINTF(LEVEL_ERROR, "class %x not supported\n",
                  pResourceRef->externalClassId);
    }
    NV_ASSERT_OR_RETURN((pClass != NULL), NV_ERR_NOT_SUPPORTED);

    if (IS_VIRTUAL(pGpu) || IS_GSP_CLIENT(pGpu))
    {
        CALL_CONTEXT *pCallContext = resservGetTlsCallContext();
        RmCtrlParams *pRmCtrlParams = pCallContext->pControlParams;
        NvHandle hClient = RES_GET_CLIENT_HANDLE(pKernelChannelGroupApi);
        NvHandle hObject = RES_GET_HANDLE(pKernelChannelGroupApi);

        NV_RM_RPC_CONTROL(pGpu,
                          hClient,
                          hObject,
                          pRmCtrlParams->cmd,
                          pRmCtrlParams->pParams,
                          pRmCtrlParams->paramsSize,
                          status);
        NV_CHECK_OR_RETURN(LEVEL_INFO, status == NV_OK, NV_ERR_NOT_SUPPORTED);
    }

    status = kchangrpSetInterleaveLevel(pGpu, pKernelChannelGroup, pParams->tsgInterleaveLevel);

    return status;
}

NV_STATUS
kchangrpapiCtrlCmdGetInterleaveLevel_IMPL
(
    KernelChannelGroupApi               *pKernelChannelGroupApi,
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS *pParams
)
{
    KernelChannelGroup *pKernelChannelGroup = NULL;
    OBJGPU *pGpu = GPU_RES_GET_GPU(pKernelChannelGroupApi);
    NvU32 subdevInst = gpumgrGetSubDeviceInstanceFromGpu(pGpu);

    if (pKernelChannelGroupApi->pKernelChannelGroup == NULL)
        return NV_ERR_INVALID_OBJECT;
    pKernelChannelGroup = pKernelChannelGroupApi->pKernelChannelGroup;

    pParams->tsgInterleaveLevel = pKernelChannelGroup->pInterleaveLevel[subdevInst];

    return NV_OK;
}

