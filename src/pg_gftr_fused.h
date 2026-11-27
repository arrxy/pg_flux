#ifndef PG_GFTR_FUSED_H
#define PG_GFTR_FUSED_H

#include "nodes/execnodes.h"

typedef struct GftrFusedGroup
{
	int32 key;
	bool key_null;
	int64 count;
	int64 sum;
	bool sum_valid;
} GftrFusedGroup;

typedef struct GftrFusedResult
{
	GftrFusedGroup *groups;
	int ngroups;
	uint64 probe_rows;
	uint64 match_rows;
	double build_ms;
	double probe_ms;
} GftrFusedResult;

/* Input attribute numbers are indexes into the join's scan slot. A zero
 * sum attribute requests COUNT(*) only. Results belong to context. */
void pg_gftr_fused_reduce(CustomScanState *join, AttrNumber group_attno,
						  AttrNumber sum_attno, MemoryContext context,
						  GftrFusedResult *result);

#endif
