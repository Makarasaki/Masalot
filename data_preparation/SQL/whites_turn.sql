UPDATE evaluations
SET WhitesTurn = CASE 
                    WHEN SUBSTR(FEN, INSTR(FEN, ' ')+1, 1) = 'w' THEN 1 
                    ELSE 0 
                 END;
