#include "postgres.h"

#include "access/xact.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type_d.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"

#include "gftr_resident.h"

PG_FUNCTION_INFO_V1(pg_gftr_resident_load);
PG_FUNCTION_INFO_V1(pg_gftr_resident_run);

typedef struct ResidentColumns
{
	int32 *first;
	int32 *second;
	int count;
	int capacity;
} ResidentColumns;

static GftrResident *resident;
static Oid resident_role;
static bool callback_registered;

static void
resident_xact(XactEvent event, void *unused)
{
	if (event == XACT_EVENT_COMMIT || event == XACT_EVENT_ABORT ||
		event == XACT_EVENT_PARALLEL_COMMIT || event == XACT_EVENT_PARALLEL_ABORT ||
		event == XACT_EVENT_PREPARE)
	{
		gftr_resident_free(resident);
		resident = NULL;
		resident_role = InvalidOid;
	}
}

static char *
resident_relation(Oid oid)
{
	char *name;

	LockRelationOid(oid, AccessShareLock);
	name = get_rel_name(oid);
	if (!name || get_rel_relkind(oid) != RELKIND_RELATION)
		ereport(ERROR, (errmsg("resident join requires ordinary tables")));
	return quote_qualified_identifier(get_namespace_name(get_rel_namespace(oid)), name);
}

static void
resident_grow(ResidentColumns *data, bool two_columns, MemoryContext context,
			  size_t *remaining)
{
	size_t row_bytes = (two_columns ? 2 : 1) * sizeof(int32);
	size_t available = *remaining / row_bytes + data->capacity;
	size_t capacity = data->capacity ? (size_t) data->capacity * 2 : 65536;

	capacity = Min(capacity, Min(available, MaxAllocSize / sizeof(int32)));
	if (capacity <= data->capacity)
		ereport(ERROR, (errmsg("resident join exceeds pg_gftr.max_input_mb")));
	*remaining -= (capacity - data->capacity) * row_bytes;
	data->first = data->first ? repalloc(data->first, capacity * sizeof(int32))
		: MemoryContextAlloc(context, capacity * sizeof(int32));
	if (two_columns)
		data->second = data->second ? repalloc(data->second, capacity * sizeof(int32))
			: MemoryContextAlloc(context, capacity * sizeof(int32));
	data->capacity = (int) capacity;
}

static ResidentColumns
resident_read(const char *sql, bool two_columns, MemoryContext context,
			  size_t *remaining)
{
	Portal portal = SPI_cursor_open_with_args(NULL, sql, 0, NULL, NULL, NULL, true, 0);
	ResidentColumns data = {0};

	if (!portal)
		elog(ERROR, "unable to open resident join input");
	for (;;)
	{
		uint64 row;
		SPITupleTable *table;

		SPI_cursor_fetch(portal, true, 65536);
		table = SPI_tuptable;
		if (!table)
			elog(ERROR, "unable to read resident join input");
		if (SPI_gettypeid(table->tupdesc, 1) != INT4OID ||
			(two_columns && SPI_gettypeid(table->tupdesc, 2) != INT4OID))
			ereport(ERROR, (errmsg("resident join input columns must have type integer")));

		for (row = 0; row < SPI_processed; row++)
		{
			HeapTuple tuple = table->vals[row];
			bool isnull;
			Datum first = SPI_getbinval(tuple, table->tupdesc, 1, &isnull);
			Datum second = (Datum) 0;

			if (isnull && !two_columns)
				continue;
			if (isnull)
				ereport(ERROR, (errmsg("resident join dimension keys cannot be NULL")));
			if (two_columns)
			{
				second = SPI_getbinval(tuple, table->tupdesc, 2, &isnull);
				if (isnull)
					ereport(ERROR, (errmsg("resident join dimension payloads cannot be NULL")));
			}
			if (data.count == data.capacity)
				resident_grow(&data, two_columns, context, remaining);
			data.first[data.count] = DatumGetInt32(first);
			if (two_columns)
				data.second[data.count] = DatumGetInt32(second);
			data.count++;
		}
		SPI_freetuptable(table);
		CHECK_FOR_INTERRUPTS();
		if (SPI_processed == 0)
			break;
	}
	SPI_cursor_close(portal);
	return data;
}

Datum
pg_gftr_resident_load(PG_FUNCTION_ARGS)
{
	char *lineitem;
	char *orders;
	char *customers;
	MemoryContext context;
	ResidentColumns fact, ord, cust;
	GftrResidentInput input;
	GftrResident *replacement = NULL;
	char error[GFTR_ERROR_SIZE] = {0};
	size_t remaining = (size_t) atoi(GetConfigOption("pg_gftr.max_input_mb", false, false)) * 1048576;
	size_t device_budget = (size_t) atoi(GetConfigOption("pg_gftr.max_device_mb", false, false)) * 1048576;
	int device = atoi(GetConfigOption("pg_gftr.device", false, false));

	if (!IsTransactionBlock() || !XactReadOnly || !IsolationUsesXactSnapshot() ||
		GetCurrentTransactionNestLevel() != 1)
		ereport(ERROR, (errmsg("resident join load requires a top-level read-only repeatable-read or serializable transaction")));
	lineitem = resident_relation(PG_GETARG_OID(0));
	orders = resident_relation(PG_GETARG_OID(1));
	customers = resident_relation(PG_GETARG_OID(2));
	context = AllocSetContextCreate(CurrentMemoryContext, "pg_gftr resident upload",
								 ALLOCSET_DEFAULT_SIZES);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "unable to connect SPI for resident join");
	fact = resident_read(psprintf("SELECT l_orderkey FROM ONLY %s", lineitem),
						 false, context, &remaining);
	ord = resident_read(psprintf("SELECT o_orderkey, o_custkey FROM ONLY %s", orders),
						true, context, &remaining);
	cust = resident_read(psprintf("SELECT c_custkey, c_nationkey FROM ONLY %s", customers),
						 true, context, &remaining);
	SPI_finish();
	input = (GftrResidentInput) {fact.first, ord.first, ord.second, cust.first, cust.second,
								fact.count, ord.count, cust.count};
	if (!callback_registered)
	{
		RegisterXactCallback(resident_xact, NULL);
		callback_registered = true;
	}
	/* Release the prior snapshot before reserving the new device budget. */
	gftr_resident_free(resident);
	resident = NULL;
	resident_role = InvalidOid;
	if (gftr_resident_load(&input, device, device_budget, &replacement, error))
		ereport(ERROR, (errmsg("pg_gftr resident join: %s", error)));
	resident = replacement;
	resident_role = GetUserId();
	MemoryContextDelete(context);
	CHECK_FOR_INTERRUPTS();
	PG_RETURN_VOID();
}

Datum
pg_gftr_resident_run(PG_FUNCTION_ARGS)
{
	char error[GFTR_ERROR_SIZE] = {0};
	int64 totals[3];
	Datum values[3];
	bool nulls[3];
	int dims[1] = {3};
	int lbs[1] = {1};

	if (!resident || resident_role != GetUserId())
		ereport(ERROR, (errmsg("load a resident join snapshot as the current role in this transaction")));
	if (gftr_resident_run(resident, totals, error))
		ereport(ERROR, (errmsg("pg_gftr resident join: %s", error)));
	CHECK_FOR_INTERRUPTS();
	for (int i = 0; i < 3; i++)
	{
		values[i] = Int64GetDatum(totals[i]);
		nulls[i] = i > 0 && totals[0] == 0;
	}
	PG_RETURN_ARRAYTYPE_P(construct_md_array(values, nulls, 1, dims, lbs,
											 INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE));
}
