import datetime
import pandas as pd

public_holidays = [
    datetime.date(2019, 1, 1),
    datetime.date(2019, 1, 28),
    datetime.date(2019, 3, 11),
    datetime.date(2019, 4, 19),
    datetime.date(2019, 4, 20),
    datetime.date(2019, 4, 21),
    datetime.date(2019, 4, 22),
    datetime.date(2019, 4, 25),
    datetime.date(2019, 6, 10),
    datetime.date(2019, 9, 27),
    datetime.date(2019, 11, 5),
    datetime.date(2019, 12, 25),
    datetime.date(2019, 12, 26)
                   ]

school_holidays = pd.date_range(start="2019-04-06", end='2019-04-22', freq='D').tolist() + \
                  pd.date_range(start="2019-06-29", end='2019-07-14', freq='D').tolist() + \
                  pd.date_range(start="2019-09-21", end='2019-10-06', freq='D').tolist() + \
                  pd.date_range(start="2019-12-21", end='2019-12-31', freq='D').tolist()

school_holidays = [x.date() for x in school_holidays]

