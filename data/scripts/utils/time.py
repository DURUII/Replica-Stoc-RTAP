import datetime


def unix_to_datetime(timestamp: str) -> datetime:
    """
    convert unix to datetime
    >>> dt = unix_to_datetime('1538402919')
    >>> print(dt.strftime('%Y/%m/%d %H:%M:%S'))
    2018/10/01 22:08:39
    """
    return datetime.datetime.fromtimestamp(int(timestamp))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
