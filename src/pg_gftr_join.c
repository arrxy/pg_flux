/* Batched inner equijoins with an int4 key and a unique build side. */
#include "postgres.h"

#include "catalog/pg_class_d.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "commands/explain.h"
#include "commands/explain_format.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/tlist.h"
#include "parser/parsetree.h"
#include "portability/instr_time.h"
#include "utils/guc.h"
#include "utils/memutils.h"

#include "gftr_join.h"

#define MIB ((size_t) 1024 * 1024)
#define JOIN_MIN_BUILD_CAPACITY 4096
#define JOIN_OUTPUT_SAMPLE_INTERVAL 1024

typedef enum JoinSide
{
	JOIN_PROBE,
	JOIN_BUILD
} JoinSide;

typedef enum JoinResultMode
{
	JOIN_RESULT_AUTO,
	JOIN_RESULT_DENSE,
	JOIN_RESULT_COMPACT
} JoinResultMode;

typedef struct JoinInput
{
	PlanState *plan;
	int key_attno;
	int max_attno;
	int ncols;
	int *attnos;
	int count;
	int capacity;
	int32 *keys;
	GftrValue **columns;
} JoinInput;

typedef struct JoinOutput
{
	JoinSide side;
	int column;					/* -1 reuses the key array */
} JoinOutput;

typedef struct GftrJoinState
{
	CustomScanState css;
	JoinInput inputs[2];
	JoinOutput *outputs;
	int noutputs;
	GftrJoinBuild *build;
	uint8 *probe_valid;
	int32 *matches;
	int batch_rows;
	int next_row;
	bool initialized;
	bool probe_done;
	bool compact;
	int device;
	size_t host_budget;
	size_t device_budget;
	size_t probe_bytes;
	MemoryContext context;
	MemoryContextCallback cleanup;
	GftrJoinTiming timing;
	double build_pack_ms;
	double probe_pack_ms;
	double output_sample_ms;
	uint64 output_samples;
	bool profile_output;
	uint64 probe_rows;
	uint64 output_rows;
	uint64 batches;
} GftrJoinState;

static bool join_enabled;
static bool join_force;
static int join_result_mode;
static int join_min_rows;
static int join_batch_rows;
static int join_max_host_mb;
static int join_max_device_mb;
static set_join_pathlist_hook_type previous_join_hook;

static const struct config_enum_entry join_result_modes[] = {
	{"auto", JOIN_RESULT_AUTO, false},
	{"dense", JOIN_RESULT_DENSE, false},
	{"compact", JOIN_RESULT_COMPACT, false},
	{NULL, 0, false}
};

static void join_paths(PlannerInfo *, RelOptInfo *, RelOptInfo *,
					   RelOptInfo *, JoinType, JoinPathExtraData *);
static Plan *join_plan(PlannerInfo *, RelOptInfo *, CustomPath *,
					   List *, List *, List *);
static Node *join_create_state(CustomScan *);
static void join_begin(CustomScanState *, EState *, int);
static TupleTableSlot *join_exec(CustomScanState *);
static void join_end(CustomScanState *);
static void join_rescan(CustomScanState *);
static void join_explain(CustomScanState *, List *, ExplainState *);

static const CustomPathMethods join_path_methods = {
	.CustomName = "GftrJoin",
	.PlanCustomPath = join_plan
};

static const CustomScanMethods join_scan_methods = {
	.CustomName = "GftrJoin",
	.CreateCustomScanState = join_create_state
};

static const CustomExecMethods join_exec_methods = {
	.CustomName = "GftrJoin",
	.BeginCustomScan = join_begin,
	.ExecCustomScan = join_exec,
	.EndCustomScan = join_end,
	.ReScanCustomScan = join_rescan,
	.ExplainCustomScan = join_explain
};

void
gftr_join_init(void)
{
	DefineCustomBoolVariable("pg_gftr.join_enabled", "Offer batched join paths.",
		NULL, &join_enabled, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_gftr.join_force", "Prefer supported join paths for experiments.",
		NULL, &join_force, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("pg_gftr.join_result_mode", "CUDA match return representation.",
		NULL, &join_result_mode, JOIN_RESULT_AUTO, join_result_modes, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.join_min_rows", "Minimum estimated probe rows unless forced.",
		NULL, &join_min_rows, 100000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.join_batch_rows", "Maximum rows per probe batch.",
		NULL, &join_batch_rows, 65536, 1, 1048576, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.join_max_host_mb", "Limit for join buffers and CPU lookup in MiB.",
		NULL, &join_max_host_mb, 256, 1, 16384, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.join_max_device_mb", "Limit for this join's device allocations in MiB.",
		NULL, &join_max_device_mb, 4096, 1, 65536, PGC_USERSET, 0, NULL, NULL, NULL);

	RegisterCustomScanMethods(&join_scan_methods);
	previous_join_hook = set_join_pathlist_hook;
	set_join_pathlist_hook = join_paths;
}

void
gftr_join_fini(void)
{
	if (set_join_pathlist_hook == join_paths)
		set_join_pathlist_hook = previous_join_hook;
}

static bool
plain_int4_var(Node *expr)
{
	Var *var;

	if (!IsA(expr, Var))
		return false;
	var = (Var *) expr;
	return var->vartype == INT4OID && var->varattno > 0 &&
		var->varlevelsup == 0 && bms_is_empty(var->varnullingrels);
}

static bool
supported_base(PlannerInfo *root, RelOptInfo *rel)
{
	RangeTblEntry *rte;

	if (rel->reloptkind != RELOPT_BASEREL || !bms_is_empty(rel->lateral_relids))
		return false;
	rte = planner_rt_fetch(rel->relid, root);
	return rte->rtekind == RTE_RELATION && rte->relkind == RELKIND_RELATION &&
		!contain_volatile_functions((Node *) rel->baserestrictinfo);
}

static void
join_paths(PlannerInfo *root, RelOptInfo *joinrel, RelOptInfo *outerrel,
		   RelOptInfo *innerrel, JoinType jointype, JoinPathExtraData *extra)
{
	Path *probe;
	Path *build;
	PlannerInfo *query;
	RestrictInfo *rinfo;
	OpExpr *op;
	Var *probe_key;
	Var *build_key;
	ListCell *lc;
	CustomPath *path;

	if (previous_join_hook)
		previous_join_hook(root, joinrel, outerrel, innerrel, jointype, extra);

	if (!join_enabled || jointype != JOIN_INNER || !extra->inner_unique ||
		list_length(extra->restrictlist) != 1 ||
		!supported_base(root, outerrel) || !supported_base(root, innerrel) ||
		!bms_is_empty(joinrel->lateral_relids))
		return;

	/* EvalPlanQual and its single-tuple rechecks require native paths. */
	for (query = root; query != NULL; query = query->parent_root)
	{
		if (query->parse->commandType != CMD_SELECT || query->parse->rowMarks ||
			query->parse->hasModifyingCTE)
			return;
	}

	rinfo = linitial_node(RestrictInfo, extra->restrictlist);
	if (rinfo->pseudoconstant || !IsA(rinfo->clause, OpExpr))
		return;

	op = (OpExpr *) rinfo->clause;
	if (op->opno != Int4EqualOperator || list_length(op->args) != 2 ||
		!plain_int4_var(linitial(op->args)) || !plain_int4_var(lsecond(op->args)))
		return;

	probe_key = linitial_node(Var, op->args);
	build_key = lsecond_node(Var, op->args);
	if (probe_key->varno == innerrel->relid)
	{
		Var *swap = probe_key;

		probe_key = build_key;
		build_key = swap;
	}
	if (probe_key->varno != outerrel->relid || build_key->varno != innerrel->relid)
		return;

	probe = outerrel->cheapest_total_path;
	build = innerrel->cheapest_total_path;
	if (!probe || !build || probe->param_info || build->param_info ||
		!list_member(probe->pathtarget->exprs, probe_key) ||
		!list_member(build->pathtarget->exprs, build_key) ||
		(!join_force && probe->rows < join_min_rows))
		return;

	foreach(lc, joinrel->reltarget->exprs)
	{
		Node *expr = lfirst(lc);

		if (!plain_int4_var(expr) ||
			(!list_member(probe->pathtarget->exprs, expr) &&
			 !list_member(build->pathtarget->exprs, expr)))
			return;
	}

	path = makeNode(CustomPath);
	path->path.pathtype = T_CustomScan;
	path->path.parent = joinrel;
	path->path.pathtarget = joinrel->reltarget;
	path->path.rows = joinrel->rows;
	path->path.disabled_nodes = build->disabled_nodes + probe->disabled_nodes;
	/* Provisional constants cover setup and packing until benchmark data is available. */
	path->path.startup_cost = build->total_cost + probe->startup_cost +
		1000.0 + build->rows * 2 * cpu_tuple_cost;
	path->path.total_cost = path->path.startup_cost +
		probe->total_cost - probe->startup_cost +
		2 * probe->rows * cpu_tuple_cost + joinrel->rows * cpu_tuple_cost;
	if (join_force)
	{
		path->path.disabled_nodes = 0;
		path->path.startup_cost = 0;
		path->path.total_cost = 0.001;
	}
	path->custom_paths = list_make2(probe, build);
	path->custom_restrictinfo = extra->restrictlist;
	/* Keep margin below the compact buffer capacity of one eighth. */
	path->custom_private = list_make3(probe_key, build_key,
		makeInteger(path->path.rows < 0.10 * probe->rows));
	path->methods = &join_path_methods;
	add_path(joinrel, &path->path);
}

static Plan *
join_plan(PlannerInfo *root, RelOptInfo *rel, CustomPath *path,
		  List *tlist, List *clauses, List *children)
{
	CustomScan *scan = makeNode(CustomScan);
	Plan *probe = linitial(children);
	Plan *build = lsecond(children);
	List *scan_tlist = NIL;
	List *outputs = NIL;
	ListCell *lc;
	TargetEntry *probe_key;
	TargetEntry *build_key;

	probe_key = tlist_member(linitial(path->custom_private), probe->targetlist);
	build_key = tlist_member(lsecond(path->custom_private), build->targetlist);
	if (!probe_key || !build_key)
		elog(ERROR, "pg_gftr join key missing from child target list");

	foreach(lc, path->path.pathtarget->exprs)
	{
		Expr *expr = lfirst(lc);
		TargetEntry *entry = tlist_member(expr, probe->targetlist);
		int side = JOIN_PROBE;

		if (!entry)
		{
			entry = tlist_member(expr, build->targetlist);
			side = JOIN_BUILD;
		}
		if (!entry)
			elog(ERROR, "pg_gftr join output missing from child target lists");

		outputs = lappend(outputs, list_make2_int(side, entry->resno));
		scan_tlist = lappend(scan_tlist, makeTargetEntry(copyObject(expr),
			list_length(scan_tlist) + 1, NULL, false));
	}

	scan->scan.plan.targetlist = tlist;
	scan->scan.scanrelid = 0;
	scan->custom_scan_tlist = scan_tlist;
	scan->custom_plans = children;
	scan->custom_private = list_make4(makeInteger(probe_key->resno),
		makeInteger(build_key->resno), outputs, lthird(path->custom_private));
	scan->methods = &join_scan_methods;
	return &scan->scan.plan;
}

static Node *
join_create_state(CustomScan *scan)
{
	GftrJoinState *state = palloc0(sizeof(*state));

	NodeSetTag(&state->css, T_CustomScanState);
	state->css.methods = &join_exec_methods;
	return (Node *) state;
}

static void
join_cleanup(void *arg)
{
	GftrJoinState *state = arg;

	gftr_join_cuda_free(state->build);
	state->build = NULL;
}

static void
join_begin(CustomScanState *node, EState *estate, int eflags)
{
	GftrJoinState *state = (GftrJoinState *) node;
	CustomScan *plan = (CustomScan *) node->ss.ps.plan;
	List *outputs = lthird(plan->custom_private);
	ListCell *lc;
	int output = 0;
	int side;

	state->context = estate->es_query_cxt;
	state->compact = join_result_mode == JOIN_RESULT_COMPACT ||
		(join_result_mode == JOIN_RESULT_AUTO && intVal(lfourth(plan->custom_private)));
	state->profile_output = estate->es_instrument != 0;
	state->device = atoi(GetConfigOption("pg_gftr.device", false, false));
	state->batch_rows = join_batch_rows;
	state->host_budget = (size_t) join_max_host_mb * MIB;
	state->device_budget = (size_t) join_max_device_mb * MIB;
	state->inputs[JOIN_PROBE].key_attno = intVal(linitial(plan->custom_private));
	state->inputs[JOIN_BUILD].key_attno = intVal(lsecond(plan->custom_private));
	state->noutputs = list_length(outputs);
	state->outputs = palloc_array(JoinOutput, state->noutputs);

	for (side = JOIN_PROBE; side <= JOIN_BUILD; side++)
	{
		JoinInput *input = &state->inputs[side];

		input->max_attno = input->key_attno;
		input->plan = ExecInitNode(list_nth(plan->custom_plans, side), estate, eflags);
		input->attnos = palloc_array(int, state->noutputs);
		input->columns = palloc0_array(GftrValue *, state->noutputs);
		node->custom_ps = lappend(node->custom_ps, input->plan);
	}

	foreach(lc, outputs)
	{
		List *source = lfirst(lc);
		int origin = linitial_int(source);
		int attno = lsecond_int(source);
		JoinInput *input = &state->inputs[origin];
		JoinOutput *target = &state->outputs[output++];

		target->side = origin;
		target->column = -1;
		if (attno != input->key_attno)
		{
			target->column = input->ncols;
			input->attnos[input->ncols++] = attno;
			input->max_attno = Max(input->max_attno, attno);
		}
	}

	state->cleanup.func = join_cleanup;
	state->cleanup.arg = state;
	MemoryContextRegisterResetCallback(state->context, &state->cleanup);
}

static void
join_host_limit(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
			 errmsg("pg_gftr join host buffer limit exceeded"),
			 errhint("Increase pg_gftr.join_max_host_mb, reduce pg_gftr.join_batch_rows, or use a native join.")));
}

static size_t
input_row_bytes(const JoinInput *input)
{
	return sizeof(int32) + (size_t) input->ncols * sizeof(GftrValue);
}

static void *
resize_buffer(MemoryContext context, void *buffer, size_t bytes)
{
	return buffer ? repalloc(buffer, bytes) : MemoryContextAlloc(context, bytes);
}

static void
resize_input(GftrJoinState *state, JoinInput *input, int capacity)
{
	int col;

	input->keys = resize_buffer(state->context, input->keys,
		(size_t) capacity * sizeof(int32));
	for (col = 0; col < input->ncols; col++)
		input->columns[col] = resize_buffer(state->context, input->columns[col],
			(size_t) capacity * sizeof(GftrValue));
	input->capacity = capacity;
}

static void
grow_build_input(GftrJoinState *state, JoinInput *build)
{
	size_t max_rows = (state->host_budget - state->probe_bytes) / input_row_bytes(build);
	size_t new_capacity;

	max_rows = Min(max_rows, MaxAllocSize / sizeof(GftrValue));
	if (build->count >= max_rows)
		join_host_limit();

	new_capacity = build->capacity > 0 ?
		(size_t) build->capacity * 2 : JOIN_MIN_BUILD_CAPACITY;
	resize_input(state, build, Min(new_capacity, max_rows));
}

static void
pack_values(JoinInput *input, TupleTableSlot *slot)
{
	int col;

	for (col = 0; col < input->ncols; col++)
	{
		int attno = input->attnos[col] - 1;
		GftrValue *value = &input->columns[col][input->count];

		value->valid = !slot->tts_isnull[attno];
		value->value = value->valid ? DatumGetInt32(slot->tts_values[attno]) : 0;
	}
}

static double
elapsed_ms(instr_time start)
{
	instr_time end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);
	return INSTR_TIME_GET_MILLISEC(end);
}

static void
load_build(GftrJoinState *state)
{
	JoinInput *build = &state->inputs[JOIN_BUILD];
	JoinInput *probe = &state->inputs[JOIN_PROBE];
	TupleTableSlot *slot;
	instr_time start;
	int status;
	char error[GFTR_ERROR_SIZE] = {0};

	state->probe_bytes = (size_t) state->batch_rows *
		(input_row_bytes(probe) + sizeof(uint8) + sizeof(int32));
	if (state->compact)
		state->probe_bytes += (size_t) state->batch_rows + sizeof(uint32);
	if (state->probe_bytes >= state->host_budget)
		join_host_limit();

	INSTR_TIME_SET_CURRENT(start);
	for (;;)
	{
		int key_idx = build->key_attno - 1;

		slot = ExecProcNode(build->plan);
		if (TupIsNull(slot))
			break;
		CHECK_FOR_INTERRUPTS();
		slot_getsomeattrs(slot, build->max_attno);
		if (slot->tts_isnull[key_idx])
			continue;
		if (build->count == build->capacity)
			grow_build_input(state, build);
		build->keys[build->count] = DatumGetInt32(slot->tts_values[key_idx]);
		pack_values(build, slot);
		build->count++;
	}
	state->build_pack_ms = elapsed_ms(start);
	state->initialized = true;
	if (build->count == 0)
	{
		state->probe_done = true;
		return;
	}

	status = gftr_join_cuda_build(build->keys, build->count, state->device,
		state->batch_rows, state->compact, state->device_budget, &state->build,
		&state->timing, error);

	CHECK_FOR_INTERRUPTS();
	if (status)
		ereport(ERROR, (errmsg("pg_gftr join: %s", error)));

	if (!state->matches)
	{
		resize_input(state, probe, state->batch_rows);
		state->probe_valid = MemoryContextAlloc(state->context,
			(size_t) state->batch_rows * sizeof(uint8));
		state->matches = MemoryContextAlloc(state->context,
			(size_t) state->batch_rows * sizeof(int32));
	}
}

static void
load_probe_batch(GftrJoinState *state)
{
	JoinInput *probe = &state->inputs[JOIN_PROBE];
	instr_time start;
	int status;
	char error[GFTR_ERROR_SIZE] = {0};

	INSTR_TIME_SET_CURRENT(start);
	probe->count = 0;
	state->next_row = 0;
	while (probe->count < state->batch_rows)
	{
		TupleTableSlot *slot = ExecProcNode(probe->plan);
		int key_idx = probe->key_attno - 1;
		bool valid;

		if (TupIsNull(slot))
		{
			state->probe_done = true;
			break;
		}
		CHECK_FOR_INTERRUPTS();
		slot_getsomeattrs(slot, probe->max_attno);
		valid = !slot->tts_isnull[key_idx];
		state->probe_valid[probe->count] = valid;
		probe->keys[probe->count] = valid ? DatumGetInt32(slot->tts_values[key_idx]) : 0;
		pack_values(probe, slot);
		probe->count++;
	}
	state->probe_pack_ms += elapsed_ms(start);

	if (probe->count == 0)
		return;

	status = gftr_join_cuda_probe(state->build, probe->keys, state->probe_valid,
		probe->count, state->matches, &state->timing, error);

	CHECK_FOR_INTERRUPTS();
	if (status)
		ereport(ERROR, (errmsg("pg_gftr join: %s", error)));

	state->probe_rows += probe->count;
	state->batches++;
}

static TupleTableSlot *
join_next(ScanState *scan)
{
	GftrJoinState *state = (GftrJoinState *) scan;
	TupleTableSlot *slot = scan->ss_ScanTupleSlot;
	JoinInput *probe = &state->inputs[JOIN_PROBE];

	ExecClearTuple(slot);
	if (!state->initialized)
		load_build(state);

	for (;;)
	{
		int probe_row;
		int build_row;
		int col;
		bool sample;
		instr_time start;

		CHECK_FOR_INTERRUPTS();
		if (state->next_row >= probe->count)
		{
			if (state->probe_done)
				return slot;
			load_probe_batch(state);
			continue;
		}

		probe_row = state->next_row++;
		build_row = state->matches[probe_row];
		if (build_row < 0)
			continue;
		if (build_row >= state->inputs[JOIN_BUILD].count)
			elog(ERROR, "pg_gftr join returned an invalid build row index");

		/* Timing every output tuple adds measurable overhead on large joins. */
		sample = state->profile_output &&
			state->output_rows % JOIN_OUTPUT_SAMPLE_INTERVAL == 0;
		if (sample)
			INSTR_TIME_SET_CURRENT(start);
		for (col = 0; col < state->noutputs; col++)
		{
			JoinOutput *source = &state->outputs[col];
			JoinInput *input = &state->inputs[source->side];
			int row = source->side == JOIN_BUILD ? build_row : probe_row;

			if (source->column < 0)
			{
				slot->tts_isnull[col] = false;
				slot->tts_values[col] = Int32GetDatum(input->keys[row]);
			}
			else
			{
				GftrValue value = input->columns[source->column][row];

				slot->tts_isnull[col] = !value.valid;
				slot->tts_values[col] = Int32GetDatum(value.value);
			}
		}

		state->output_rows++;
		ExecStoreVirtualTuple(slot);
		if (sample)
		{
			state->output_sample_ms += elapsed_ms(start);
			state->output_samples++;
		}
		return slot;
	}
}

static bool
join_recheck(ScanState *scan, TupleTableSlot *slot)
{
	elog(ERROR, "pg_gftr join does not support EvalPlanQual");
	return false;
}

static TupleTableSlot *
join_exec(CustomScanState *node)
{
	if (node->ss.ps.state->es_epq_active)
		elog(ERROR, "pg_gftr join does not support EvalPlanQual");
	return ExecScan(&node->ss, join_next, join_recheck);
}

static void
join_end(CustomScanState *node)
{
	GftrJoinState *state = (GftrJoinState *) node;

	join_cleanup(state);
	ExecEndNode(state->inputs[JOIN_PROBE].plan);
	ExecEndNode(state->inputs[JOIN_BUILD].plan);
}

static void
join_rescan(CustomScanState *node)
{
	GftrJoinState *state = (GftrJoinState *) node;
	int side;

	join_cleanup(state);
	ExecScanReScan(&node->ss);
	state->initialized = false;
	state->probe_done = false;
	state->next_row = 0;
	state->probe_rows = 0;
	state->output_rows = 0;
	state->batches = 0;
	state->build_pack_ms = 0;
	state->probe_pack_ms = 0;
	state->output_sample_ms = 0;
	state->output_samples = 0;
	memset(&state->timing, 0, sizeof(state->timing));

	for (side = JOIN_PROBE; side <= JOIN_BUILD; side++)
	{
		JoinInput *input = &state->inputs[side];

		input->count = 0;
		if (node->ss.ps.chgParam)
			UpdateChangedParamSet(input->plan, node->ss.ps.chgParam);
		if (!input->plan->chgParam)
			ExecReScan(input->plan);
	}
}

static void
join_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GftrJoinState *state = (GftrJoinState *) node;

	ExplainPropertyText("Backend", "CUDA", es);
	ExplainPropertyBool("Compaction Requested", state->compact, es);
	ExplainPropertyInteger("Probe Batch Size", NULL, state->batch_rows, es);
	ExplainPropertyInteger("Projected Columns", NULL, state->noutputs, es);

	if (es->analyze && state->initialized)
	{
		if (state->timing.algorithm)
			ExplainPropertyText("Algorithm", state->timing.algorithm, es);
		ExplainPropertyInteger("Build Rows", NULL, state->inputs[JOIN_BUILD].count, es);
		ExplainPropertyUInteger("Probe Rows", NULL, state->probe_rows, es);
		ExplainPropertyUInteger("Output Rows", NULL, state->output_rows, es);
		ExplainPropertyUInteger("Probe Batches", NULL, state->batches, es);
		ExplainPropertyUInteger("Compact Batches", NULL, state->timing.compact_batches, es);
		ExplainPropertyUInteger("Dense Batches", NULL, state->timing.dense_batches, es);
		ExplainPropertyUInteger("Build Bytes", NULL, state->timing.build_bytes, es);
		ExplainPropertyUInteger("H2D Bytes", NULL, state->timing.h2d_bytes, es);
		ExplainPropertyUInteger("D2H Bytes", NULL, state->timing.d2h_bytes, es);
		ExplainPropertyFloat("Build Scan and Pack", "ms", state->build_pack_ms, 3, es);
		ExplainPropertyFloat("Probe Scan and Pack", "ms", state->probe_pack_ms, 3, es);
		if (state->output_samples)
			ExplainPropertyFloat("Estimated Output Materialization", "ms",
				state->output_sample_ms * state->output_rows / state->output_samples, 3, es);
		ExplainPropertyFloat("Build", "ms", state->timing.build_ms, 3, es);
		ExplainPropertyFloat("H2D", "ms", state->timing.h2d_ms, 3, es);
		ExplainPropertyFloat("Probe", "ms", state->timing.probe_ms, 3, es);
		ExplainPropertyFloat("D2H", "ms", state->timing.d2h_ms, 3, es);
	}
}
