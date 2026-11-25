"""Packing a step's samples into per-rank micro batches, by reference.

The orchestrator never holds a sample's tokens: it packs ``SampleRef``s by token count into
``MicroBatchPlan``s, and each trainer rank builds its micro batches from the plans
(``prime_rl.transports.batch.materialize``).
"""

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import msgspec
import numpy as np

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import get_model
from prime_rl.trainer.perf import FlopCounter
from prime_rl.transports.batch.types import MicroBatchPlan, SampleRef


@dataclass
class _WeightedSet:
    total: int = 0
    items: list[int] = field(default_factory=list)

    def add(self, idx: int, weight: int) -> None:
        self.items.append(idx)
        self.total += weight

    def merge(self, other: "_WeightedSet") -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: "_WeightedSet") -> bool:
        if self.total != other.total:
            return self.total < other.total
        return self.items < other.items


class _KKState:
    def __init__(self, items: list[tuple[int, int]], k: int):
        self.sets = [_WeightedSet() for _ in range(k)]
        for set_idx, (idx, weight) in enumerate(items):
            self.sets[set_idx].add(idx, weight)
        self.sets.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].total - self.sets[-1].total

    def merge(self, other: "_KKState") -> None:
        k = len(self.sets)
        for i in range(k):
            self.sets[i].merge(other.sets[k - 1 - i])
        self.sets.sort(reverse=True)

    def partitions(self) -> list[list[int]]:
        return [sorted(weighted_set.items) for weighted_set in self.sets]

    def __lt__(self, other: "_KKState") -> bool:
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]


def _karmarkar_karp(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    assert len(weights) >= num_partitions
    assert len(weights) % num_partitions == 0
    weighted_indices = sorted((weight, idx) for idx, weight in enumerate(weights))
    states: list[_KKState] = []
    for offset in range(0, len(weighted_indices), num_partitions):
        items = [(idx, weight) for weight, idx in weighted_indices[offset : offset + num_partitions]]
        heapq.heappush(states, _KKState(items, num_partitions))

    while len(states) > 1:
        state = heapq.heappop(states)
        state.merge(heapq.heappop(states))
        heapq.heappush(states, state)

    return states[0].partitions()


def _partition_loads(weights: Sequence[int], partitions: list[list[int]]) -> list[int]:
    return [sum(weights[i] for i in partition) for partition in partitions]


_NO_OTHER_MAX, _NO_OTHER_MIN = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def _others_extreme(
    relative: np.ndarray, ranked: np.ndarray, rank: int, partners: np.ndarray, empty: int
) -> np.ndarray:
    """For each partner, the first load in ``ranked`` order of a rank other than ``rank`` and that
    partner (``empty`` when there is none): the max (or min) over the ranks a swap leaves alone."""
    leading = ranked[:3]
    allowed = (leading != rank) & (leading[None, :] != partners[:, None])
    first = allowed.argmax(axis=1)
    return np.where(allowed.any(axis=1), relative[leading[first]], empty)


def _refine_by_swapping(weights: Sequence[int], partitions: list[list[int]]) -> list[list[int]]:
    """Apply the best pairwise swap until none lowers ``(max load, max - min load)``.

    Ties go to the first swap in (rank pair, left position, right position) order. A swap
    between two ranks that are neither at the max nor at the min load leaves both, so each
    round scores only those ranks' swaps, with every other rank at once. Loads are taken
    relative to the current max load, so the int64 math stays within the spread plus the
    largest weight even when total FLOPs exceed int64.
    """
    num_partitions = len(partitions)
    if num_partitions < 2:
        return [list(partition) for partition in partitions]
    assert len({len(partition) for partition in partitions}) == 1, "Partitions must be the same size"
    items = np.array(partitions, dtype=np.int64)
    item_weights = np.asarray(weights, dtype=np.int64)[items]
    loads = _partition_loads(weights, partitions)

    while True:
        top, bottom = max(loads), min(loads)
        relative = np.array([load - top for load in loads], dtype=np.int64)
        ascending = np.argsort(relative, kind="stable")
        descending = ascending[::-1]
        # (new max, spread, rank pair, left position, right position) of the best swap so far
        best: tuple[int, ...] | None = None
        for rank in (rank for rank in range(num_partitions) if loads[rank] in (top, bottom)):
            partners = np.delete(np.arange(num_partitions), rank)
            other_max = _others_extreme(relative, descending, rank, partners, _NO_OTHER_MAX)[:, None, None]
            other_min = _others_extreme(relative, ascending, rank, partners, _NO_OTHER_MIN)[:, None, None]
            # [partner, position in rank, position in partner]
            own = item_weights[rank][None, :, None]
            theirs = item_weights[partners][:, None, :]
            new_own = relative[rank] - own + theirs
            new_theirs = relative[partners][:, None, None] - theirs + own
            new_max = np.maximum(np.maximum(new_own, new_theirs), other_max)
            lowest_max = new_max.min()
            tied = new_max == lowest_max
            spread = np.where(tied, new_max - np.minimum(np.minimum(new_own, new_theirs), other_min), _NO_OTHER_MIN)
            least_spread = spread.min()
            for partner_index, own_pos, their_pos in np.argwhere(spread == least_spread).tolist():
                partner = int(partners[partner_index])
                if rank < partner:
                    swap = (int(lowest_max), int(least_spread), rank, partner, own_pos, their_pos)
                else:
                    swap = (int(lowest_max), int(least_spread), partner, rank, their_pos, own_pos)
                if best is None or swap < best:
                    best = swap
        if best is None or best[:2] >= (0, top - bottom):
            return items.tolist()

        _, _, left_rank, right_rank, left_pos, right_pos = best
        left_weight, right_weight = int(item_weights[left_rank, left_pos]), int(item_weights[right_rank, right_pos])
        items[left_rank, left_pos], items[right_rank, right_pos] = (
            items[right_rank, right_pos],
            items[left_rank, left_pos],
        )
        item_weights[left_rank, left_pos], item_weights[right_rank, right_pos] = right_weight, left_weight
        loads[left_rank] += right_weight - left_weight
        loads[right_rank] += left_weight - right_weight


def balanced_partition(weights: Sequence[int], num_partitions: int) -> list[list[int]]:
    """Partition item indices into ``num_partitions`` groups of near-equal total weight.

    Requires ``len(weights)`` to be a positive multiple of ``num_partitions``.
    """
    partitions = _karmarkar_karp(weights, num_partitions)
    return _refine_by_swapping(weights, partitions)


def _has_routing(ref: SampleRef) -> bool:
    """Whether the sample's body holds routing rows."""
    return "routed_experts" in ref.fields


@dataclass
class _Bin:
    samples: list[SampleRef] = field(default_factory=list)
    length: int = 0

    def add(self, sample: SampleRef) -> None:
        self.samples.append(sample)
        self.length += sample.num_tokens

    def workload(self, bin_cost: Callable[[Sequence[int]], int]) -> int:
        return bin_cost([sample.num_tokens for sample in self.samples])

    def split_by_workload(self, bin_cost: Callable[[Sequence[int]], int]) -> tuple["_Bin", "_Bin"]:
        # Greedily place the heaviest sample on the currently lighter side (longest-processing-time).
        ranked = sorted(self.samples, key=lambda sample: -bin_cost([sample.num_tokens]))
        left, right = _Bin(), _Bin()
        left_workload = right_workload = 0
        for sample in ranked:
            sample_workload = bin_cost([sample.num_tokens])
            if left_workload <= right_workload:
                left.add(sample)
                left_workload += sample_workload
            else:
                right.add(sample)
                right_workload += sample_workload
        return left, right


def _expand_bins_by_splitting(bins: list[_Bin], target_count: int, bin_cost: Callable[[Sequence[int]], int]) -> None:
    while len(bins) < target_count:
        candidates = [
            (bin_content.workload(bin_cost), idx)
            for idx, bin_content in enumerate(bins)
            if len(bin_content.samples) > 1
        ]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = bins[idx].split_by_workload(bin_cost)
        bins[idx] = left
        bins.append(right)


def pack_samples(
    samples: list[SampleRef],
    max_seq_len: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
) -> list[_Bin]:
    """Pack samples into micro batches with First Fit Decreasing, minimizing padding while never
    truncating; with several data ranks, split the heaviest bins until every rank gets one.
    Samples of any temperature, and multimodal samples with text ones, pack together."""
    bins: list[_Bin] = []
    # Loss routing is per token (component weight streams), so samples of different loss types
    # and raw-image refs pack together freely. Routing replays per micro batch: every sample in
    # a bin has routing rows or none does, so a sample's first fit is the first bin of its
    # routing class with room.
    classes: dict[bool, list[_Bin]] = {False: [], True: []}
    room = {routing: np.zeros(len(samples), dtype=np.int64) for routing in classes}
    for sample in sorted(samples, key=lambda sample: -sample.num_tokens):
        routing = _has_routing(sample)
        candidates = classes[routing]
        fits = room[routing][: len(candidates)] >= sample.num_tokens
        first = int(fits.argmax()) if len(candidates) else 0
        if len(candidates) and fits[first]:
            candidates[first].add(sample)
            room[routing][first] -= sample.num_tokens
        else:
            room[routing][len(candidates)] = max_seq_len - sample.num_tokens
            candidates.append(_Bin([sample], sample.num_tokens))
            bins.append(candidates[-1])

    if num_train_workers > 1:
        target_count = max(
            ((len(bins) + num_train_workers - 1) // num_train_workers) * num_train_workers,
            num_train_workers,
        )
        _expand_bins_by_splitting(bins, target_count, bin_cost)
    return bins


def sequence_lengths(plan: MicroBatchPlan) -> list[int]:
    """The plan's sequence lengths as the trainer builds them: padding joins the last sequence."""
    lengths = [sample.num_tokens for sample in plan.samples]
    lengths[-1] += plan.padding
    return lengths


def _distribute_group(
    group: list[MicroBatchPlan],
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
) -> list[list[MicroBatchPlan]]:
    # Callers pad each group to a positive multiple of num_train_workers first.
    assert len(group) % num_train_workers == 0, "Number of micro batches is not divisible by number of data ranks"
    if not group:
        return [[] for _ in range(num_train_workers)]

    weights = [bin_cost(sequence_lengths(plan)) for plan in group]
    partitions = balanced_partition(weights, num_train_workers)
    return [[group[i] for i in partition] for partition in partitions]


def _pad_group_for_distribution(group: list[MicroBatchPlan], num_train_workers: int) -> list[MicroBatchPlan]:
    """Pad a group of micro batches with zero-loss copies of its first so its length is
    divisible by num_train_workers."""
    num_padding = -len(group) % num_train_workers
    if num_padding > 0 and len(group) > 0:
        dummy = msgspec.structs.replace(group[0], dummy=True)
        group.extend([dummy] * num_padding)
    return group


def prepare_batch(
    samples: list[SampleRef],
    seq_len: int,
    num_train_workers: int,
    bin_cost: Callable[[Sequence[int]], int],
    *,
    trained_version: int,
    min_version: int,
    pad_to_multiple_of: int = 1,
) -> list[list[MicroBatchPlan]]:
    """Plan a step's micro batches for each data rank. Each micro batch is one row of at most
    ``seq_len`` tokens holding any number of samples.

    FSDP requires all ranks to execute the same operations at each step. If one rank
    processes a multimodal batch (triggering the vision encoder) while another processes
    a text-only batch, the all-gather will hang. We separate micro batches by modality
    and distribute them so that at each step index, all ranks see the same modality.
    """
    plans = []
    for bin_content in pack_samples(samples, seq_len, num_train_workers, bin_cost):
        padding = (pad_to_multiple_of - bin_content.length % pad_to_multiple_of) % pad_to_multiple_of
        plans.append(
            MicroBatchPlan(
                samples=bin_content.samples,
                trained_version=trained_version,
                min_version=min_version,
                padding=padding if pad_to_multiple_of > 1 else 0,
            )
        )

    # Separate by modality so each step index has uniform modality across all ranks
    mm_plans = [plan for plan in plans if any(sample.images for sample in plan.samples)]
    text_plans = [plan for plan in plans if not any(sample.images for sample in plan.samples)]

    # Pad each group independently so its count is divisible by num_train_workers
    mm_plans = _pad_group_for_distribution(mm_plans, num_train_workers)
    text_plans = _pad_group_for_distribution(text_plans, num_train_workers)

    plans_per_gpu: list[list[MicroBatchPlan]] = [[] for _ in range(num_train_workers)]
    for group in (mm_plans, text_plans):
        for worker_idx, worker_plans in enumerate(_distribute_group(group, num_train_workers, bin_cost)):
            plans_per_gpu[worker_idx].extend(worker_plans)
    return plans_per_gpu


class BatchPacker:
    """Bin-packs a step's training samples into one micro-batch plan list per trainer DP rank."""

    def __init__(self, config: OrchestratorConfig):
        self.seq_len = config.seq_len
        self.num_train_workers = config.num_train_workers
        # Rows hold at most `seq_len` tokens, so padding to a multiple of it pads every row to exactly `seq_len`
        self.pad_to_multiple_of = config.seq_len if config.pad_to_seq_len else config.pad_to_multiple_of
        # Built as the trainer builds it. The flash-attention version selects kernels, not modules.
        model_config = ModelConfig(
            name=config.model.name, attn="flash_attention_3", trust_remote_code=config.tokenizer.trust_remote_code
        )
        self.bin_cost = FlopCounter(get_model(model_config))

    def pack(self, samples: list[SampleRef], *, trained_version: int, min_version: int) -> list[list[MicroBatchPlan]]:
        return prepare_batch(
            samples,
            seq_len=self.seq_len,
            num_train_workers=self.num_train_workers,
            bin_cost=self.bin_cost,
            trained_version=trained_version,
            min_version=min_version,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
