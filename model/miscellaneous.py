from datetime import datetime, timedelta


def generate_date_range(start: str, step: int, end: str = None):
    if end is None:
        end = datetime.now().date()
    else:
        end = datetime.strptime(end, '%Y-%m-%d').date()
    start = datetime.strptime(start, '%Y-%m-%d').date()

    f = end.replace(year=end.year - step)
    t = end.replace()
    while f > start:
        print(f.strftime('%Y-%m-%d'))
        print((t - timedelta(days=1)).strftime('%Y-%m-%d'))
        f = f.replace(year=f.year - step)
        t = t.replace(year=t.year - step)

        while True:
            i = input('Continue? ENTER')
            if i == '':
                print('\n' * 100)
                break
    print(start.strftime('%Y-%m-%d'))
    print((t - timedelta(days=1)).strftime('%Y-%m-%d'))


if __name__ == '__main__':
    generate_date_range('1994-08-31', 3, '2024-01-02')
