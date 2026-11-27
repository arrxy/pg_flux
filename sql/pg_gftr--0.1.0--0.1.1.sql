\echo Use "ALTER EXTENSION pg_gftr UPDATE" to load this file. \quit

CREATE FUNCTION pg_gftr_resident_load(regclass, regclass, regclass) RETURNS void
AS 'MODULE_PATHNAME', 'pg_gftr_resident_load'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION pg_gftr_resident_run() RETURNS bigint[]
AS 'MODULE_PATHNAME', 'pg_gftr_resident_run'
LANGUAGE C VOLATILE PARALLEL UNSAFE;
