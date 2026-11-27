#ifndef GFTR_RESIDENT_H
#define GFTR_RESIDENT_H

#include "gftr_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GftrResident GftrResident;
typedef struct GftrResidentInput
{
    const int32_t *facts;
    const int32_t *orders;
    const int32_t *order_customers;
    const int32_t *customers;
    const int32_t *nations;
    int32_t fact_count;
    int32_t order_count;
    int32_t customer_count;
} GftrResidentInput;

int gftr_resident_load(const GftrResidentInput *input, int device, size_t budget,
                       GftrResident **out, char error[GFTR_ERROR_SIZE]);
int gftr_resident_run(GftrResident *state, int64_t result[3], char error[GFTR_ERROR_SIZE]);
void gftr_resident_free(GftrResident *state);

#ifdef __cplusplus
}
#endif

#endif
