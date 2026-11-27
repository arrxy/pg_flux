#include "gftr_resident.h"
#include <stdio.h>

int
gftr_resident_load(const GftrResidentInput *input, int device, size_t budget,
				   GftrResident **out, char error[GFTR_ERROR_SIZE])
{
	(void) input;
	(void) device;
	(void) budget;
	*out = NULL;
	snprintf(error, GFTR_ERROR_SIZE, "resident joins require a CUDA build");
	return -1;
}

int
gftr_resident_run(GftrResident *state, int64_t result[3], char error[GFTR_ERROR_SIZE])
{
	(void) state;
	(void) result;
	snprintf(error, GFTR_ERROR_SIZE, "resident joins require a CUDA build");
	return -1;
}

void
gftr_resident_free(GftrResident *state)
{
	(void) state;
}
