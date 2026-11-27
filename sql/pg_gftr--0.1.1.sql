\echo Use "CREATE EXTENSION pg_gftr" to load this file. \quit

CREATE FUNCTION pg_gftr_version() RETURNS text
AS 'MODULE_PATHNAME', 'pg_gftr_version'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

-- Loading the C function registers the planner hook in the creating session.
SELECT pg_gftr_version();

CREATE FUNCTION pg_gftr_resident_load(regclass, regclass, regclass) RETURNS void
AS 'MODULE_PATHNAME', 'pg_gftr_resident_load'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION pg_gftr_resident_run() RETURNS bigint[]
AS 'MODULE_PATHNAME', 'pg_gftr_resident_run'
LANGUAGE C VOLATILE PARALLEL UNSAFE;
