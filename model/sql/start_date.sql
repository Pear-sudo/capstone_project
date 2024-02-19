with start as (select min(trddt) as start_date from stock group by stkcd)
select max(start_date)
from start