-- With read_bytea_embbeding.sql you create a stored function to read bytea value from embbeding table and then it read 100
-- This query use the stored function to read an exact list of item_id specified in the WHERE

WITH unnested_floats AS (
  SELECT
    e.item_id,
    public.bytea_to_float4_sql(substring(e.embedding FROM s.byte_offset + 1 FOR 4)) AS val,
    s.byte_offset
  FROM (
    SELECT item_id, embedding
    FROM public.embedding
    WHERE item_id IN (
      '820a6ead201e2cc7e010d96ffbc6620c',
      '5dd8b5d591d6cd8570ad67a9f2a2f27e',
      '1905b7448eba3761ec21b78090e5b31b',
      'a3ebd2d9ca1f8ad7a94fa91133d3d56b',
      '0f1dc3ef5d0a2dbda6f44fd316a9f4f9',
      'aea3b3911309c25b1f285efaad62493b',
      'c8ab6751ea6fdfbb8d89602caae4176b',
      '61c248ce786f17709d7f2f24209999c2',
      '307d090ec020ff7d4b9395f282e11bdb',
      'fca5d97c102adda841fac26821485beb',
      'b6f4e515360af0def3f04cb6a1d3bde3',
      '596d0787c92a761c0b2658f323cd90da',
      '70b190e690171f571f0ed20119d360bf',
      '6e3f3ffd06a3a675b15ffda425c808c2',
      '3d0665f7cedf42f8f24aca4b16c0582f',
      '9934be9305d5c3a2d67eb69220cab204',
      '5e15340a3e1b94e841a8619b1426a78b',
      '04bd0874d150a22172ee2a75d5f45abe',
      'da215a422754833b1e956140fb196d9a',
      'b77b069fcb1586389b902ce3c8d310f6',
      '33956b286c2145076bbe31560b8d6243',
      '1804efecb97d5da6ef8bf564d7b24d0e',
      'ebca206df77d49975425c100ab285c6f',
      'cdb651159553caeb535775fab0b893ce',
      '4e9099f9d5f828e01be7eca8dfed035b'
    )
  ) AS e
  CROSS JOIN LATERAL generate_series(0, length(e.embedding) - 4, 4) AS s(byte_offset)
)
SELECT
  item_id,
  to_jsonb(array_agg(val ORDER BY byte_offset)) AS embedding_json
FROM unnested_floats
GROUP BY item_id;