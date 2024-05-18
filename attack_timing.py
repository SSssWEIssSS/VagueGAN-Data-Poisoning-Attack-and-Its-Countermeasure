from federated_learning.utils import default_no_change

from federated_learning.worker_selection import BeforeBreakpoint
from federated_learning.worker_selection import AfterBreakpoint
from server import run_exp

if __name__ == '__main__':
    START_EXP_IDX = 0
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 25
    REPLACEMENT_METHOD = default_no_change
    KWARGS = {
        "BeforeBreakPoint_EPOCH" : 75,
        "BeforeBreakpoint_NUM_WORKERS_PER_ROUND" : 5,
        "AfterBreakPoint_EPOCH" : 75,
        "AfterBreakpoint_NUM_WORKERS_PER_ROUND" : 5,
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, AfterBreakpoint(), experiment_id)
