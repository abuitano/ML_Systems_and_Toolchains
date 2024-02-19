UPDATE rdata
SET moment = '2020-03-10'::date
WHERE id < 2 OR
id > 4;

SELECT * FROM rdata;