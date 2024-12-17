UPDATE evaluations
SET 
    castling_KW = CASE WHEN INSTR(
                           SUBSTR(FEN, INSTR(FEN,' ')+3, INSTR(SUBSTR(FEN, INSTR(FEN,' ')+3), ' ') - 1),
                           'K') > 0 THEN 1 ELSE 0 END,
    castling_QW = CASE WHEN INSTR(
                           SUBSTR(FEN, INSTR(FEN,' ')+3, INSTR(SUBSTR(FEN, INSTR(FEN,' ')+3), ' ') - 1),
                           'Q') > 0 THEN 1 ELSE 0 END,
    castling_kb = CASE WHEN INSTR(
                           SUBSTR(FEN, INSTR(FEN,' ')+3, INSTR(SUBSTR(FEN, INSTR(FEN,' ')+3), ' ') - 1),
                           'k') > 0 THEN 1 ELSE 0 END,
    castling_qb = CASE WHEN INSTR(
                           SUBSTR(FEN, INSTR(FEN,' ')+3, INSTR(SUBSTR(FEN, INSTR(FEN,' ')+3), ' ') - 1),
                           'q') > 0 THEN 1 ELSE 0 END;
