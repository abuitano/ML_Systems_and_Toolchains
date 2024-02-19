CREATE TABLE rdata (
id SERIAL PRIMARY KEY,
a TEXT CHECK (char_length(a) <= 5),
b TEXT CHECK (char_length(b) <= 5),
moment DATE,
x numeric(5,2)
);