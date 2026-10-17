"""Small, reusable replica recommendation policy."""
from math import ceil

CAPACITY_PER_REPLICA_RPS = 60.0
MIN_REPLICAS = 1
MAX_REPLICAS = 10


def recommend_replicas(request_rate_rps, capacity_per_replica=CAPACITY_PER_REPLICA_RPS,
                       minimum=MIN_REPLICAS, maximum=MAX_REPLICAS):
    if capacity_per_replica <= 0 or minimum < 0 or maximum < minimum:
        raise ValueError("Invalid capacity or replica bounds")
    target = ceil(max(0.0, float(request_rate_rps)) / capacity_per_replica)
    return max(minimum, min(maximum, target))


def update_replicas(current, target, max_step=1):
    if max_step < 0:
        raise ValueError("max_step must be non-negative")
    if target > current:
        return min(target, current + max_step)
    if target < current:
        return max(target, current - max_step)
    return current
