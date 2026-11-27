\set ON_ERROR_STOP on

CREATE TEMP TABLE resident_fact(l_orderkey integer);
CREATE TEMP TABLE resident_orders(o_orderkey integer, o_custkey integer);
CREATE TEMP TABLE resident_customer(c_custkey integer, c_nationkey integer);
INSERT INTO resident_fact VALUES (1), (1), (2), (9), (NULL), (-1), (3);
INSERT INTO resident_orders VALUES (1, 6), (2, 7), (3, 100);
INSERT INTO resident_customer VALUES (6, 0), (7, 2);

BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT pg_gftr_resident_load('resident_fact', 'resident_orders', 'resident_customer');
DO $$
DECLARE
    expected bigint[];
BEGIN
    SELECT ARRAY[count(*), sum(o.o_custkey), sum(c.c_nationkey)]
      INTO expected
      FROM resident_fact l JOIN resident_orders o ON o.o_orderkey = l.l_orderkey
      JOIN resident_customer c ON c.c_custkey = o.o_custkey;
    IF pg_gftr_resident_run() IS DISTINCT FROM expected
       OR pg_gftr_resident_run() IS DISTINCT FROM expected
    THEN
        RAISE EXCEPTION 'resident join result mismatch';
    END IF;
END $$;
COMMIT;

DO $$
BEGIN
    BEGIN
        PERFORM pg_gftr_resident_run();
        RAISE EXCEPTION 'resident join survived commit';
    EXCEPTION WHEN internal_error THEN
        IF SQLERRM NOT LIKE 'load a resident join snapshot%' THEN RAISE; END IF;
    END;
END $$;

TRUNCATE resident_customer;
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT pg_gftr_resident_load('resident_fact', 'resident_orders', 'resident_customer');
DO $$
BEGIN
    IF pg_gftr_resident_run() IS DISTINCT FROM ARRAY[0, NULL, NULL]::bigint[]
    THEN
        RAISE EXCEPTION 'empty resident join must preserve NULL SUM results';
    END IF;
END $$;
ROLLBACK;

DO $$
BEGIN
    BEGIN
        PERFORM pg_gftr_resident_run();
        RAISE EXCEPTION 'resident join survived rollback';
    EXCEPTION WHEN internal_error THEN
        IF SQLERRM NOT LIKE 'load a resident join snapshot%' THEN RAISE; END IF;
    END;
END $$;
