# Deterministic scheduling and static tensor-memory planning

## Scope

Milestone 7 builds an execution order and a static tensor-storage plan from a fully loaded and
semantically verified ForgeIR graph. It does not evaluate an operation, allocate the described arena,
copy parameter data, or measure process, CPU, or GPU memory.

The planner consumes explicit graph schema `1.0` shapes and dtypes. Dynamic dimensions are not
resolved by this milestone; a nonpositive dimension fails planning. All byte counts are descriptor
calculations and all allocation decisions are recorded as data.

## Execution schedule

`build_execution_schedule` constructs a fresh deterministic topological order rather than treating
the loader's structural ordering as an execution schedule. Data-dependency edges come from each
operation input to the operation that produced it. Side-effect-marked operations receive additional
edges that preserve their relative graph order. Kahn ordering selects the lexicographically smallest
ready operation ID, making independent-operation choices deterministic.

Every schedule entry contains a zero-based index, stable operation ID, semantic name, canonical type,
ordered inputs, and ordered outputs. Input, Parameter, and Constant declaration operations remain in
the schedule because their definition indices establish boundary-value lifetimes. A missing producer,
duplicate producer, cycle, or side-effect ordering conflict fails explicitly.

## Lifetime model

For every graph value, the planner records:

- `definition_index`: schedule index of its producing operation;
- `first_use_index`: earliest scheduled consumer, or null when no graph operation consumes it;
- `final_use_index`: latest scheduled consumer;
- inclusive lifetime interval `[definition_index, final_use_index]`;
- required alignment, raw byte size, aligned byte size, storage class, and optional arena offset.

An input read and output write at the same operation index are considered simultaneously live. A
reusable allocation is therefore released only when `final_use_index < next_definition_index`, never
when the indices are equal.

Declared outputs use the one-past-final-operation index as their final-use boundary. Their buffers
remain protected after the schedule completes. An unused intermediate has incomplete lifetime
information and is rejected; optimization should remove such dead values before planning.

## Storage ownership

The plan has four explicit storage classes:

- `external_input`: caller-owned, protected, and never assigned an arena offset;
- `external_immutable`: Parameter and Constant storage, explicitly immutable and external to the
  reusable arena;
- `arena_reusable`: intermediate values eligible for reuse only after their inclusive lifetime ends;
- `arena_output`: dedicated aligned storage for declared outputs, never selected from a reusable free
  block and retained through the output boundary.

Parameters already resolve to external NPZ weights under the graph contract. Constant literals are
also classified as immutable external plan data. Neither class contributes an offset or free block.
Input storage is never repurposed, so an input remains protected through its recorded final use.

## Size and alignment contract

The CPU default alignment is 64 bytes. `MemoryPlannerOptions` carries both a backend name and an
alignment override so a future backend can select its required boundary without changing the
algorithm. Alignment must be a positive power of two.

`TensorDescriptor::byte_size` checks element-count and dtype-width multiplication. The planner checks
alignment rounding, naive-byte accumulation, arena growth, free-block splitting/coalescing, live-byte
accumulation, and allocation-range endpoints before arithmetic. Scalars have one element. Zero or
negative/dynamic dimensions, byte-size overflow, allocation overflow, and missing lifetime data fail
without producing a partial plan.

## Deterministic best-fit arena

Arena allocation processes managed values by definition index and then stable value ID. Before each
definition it releases reusable allocations whose final use is strictly earlier and coalesces adjacent
free blocks. For an intermediate, deterministic best fit selects:

1. the smallest free block that contains the aligned allocation;
2. the lowest arena offset when block sizes tie.

The selected block is consumed from its low address and any aligned remainder stays free. If no block
fits, the arena grows with checked addition. Declared outputs always grow the arena and are never
placed in a previously used free block.

Plan verification rejects a misaligned or out-of-range allocation, missing arena offset, protected
value with reusable storage, any overlapping memory ranges for simultaneously live values, or any
range reuse involving an output allocation.

## Calculated memory summaries

External inputs, parameters, and constants are excluded from arena comparisons because the planner
does not own them. For managed intermediates and outputs:

- `naive_allocation_bytes` is the sum of each independently aligned allocation;
- `planned_bytes` and `arena_size_bytes` are the arena high-water extent after reuse and protected
  output placement;
- `peak_live_bytes` is the maximum sum of aligned sizes whose inclusive lifetimes cover one schedule
  index;
- `reuse_ratio` is `naive_allocation_bytes / planned_bytes`;
- `reuse_fraction` is `(naive_allocation_bytes - planned_bytes) / naive_allocation_bytes`.

These values are deterministic static calculations from graph metadata. They are not allocator
measurements, device-memory observations, benchmark results, or performance claims.

## CLI and artifacts

The required form is:

    forgeir_cli plan-memory <graph>

The default output directory is `<graph-parent>/<graph-stem>.memory_plan`. Optional arguments select
an alignment or explicit destination:

    forgeir_cli plan-memory <graph> --alignment <bytes> --output-dir <directory>

The CLI loads and semantically verifies the graph before planning, then writes:

- `schedule.json`: the ordered operation schedule;
- `memory_plan.json`: complete lifetimes, storage classes, offsets, and calculated summaries;
- `timeline.csv`: one deterministic row per graph value;
- `timeline.svg`: an operation-index versus arena-offset visualization generated directly from the
  plan's inclusive lifetimes and aligned allocation ranges.

The CLI prints the graph hash, backend, alignment, operation count, planned and peak-live bytes, naive
bytes, both reuse calculations, and all artifact paths. Artifact JSON uses stable object ordering and
a final newline; repeated plans produce byte-identical outputs.
