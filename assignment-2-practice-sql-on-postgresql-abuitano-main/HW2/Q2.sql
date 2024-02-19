DROP TABLE rdata;

CREATE TABLE rdata (
id SERIAL PRIMARY KEY,
a TEXT UNIQUE NOT NULL CHECK (char_length(a) <= 5),
b TEXT UNIQUE NOT NULL CHECK (char_length(b) <= 5),
moment DATE DEFAULT '2020-01-01'::DATE,
x numeric(5,2) CHECK (x > 0)
);