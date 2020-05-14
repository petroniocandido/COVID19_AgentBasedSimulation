def number_of_days(iteration):
    return iteration // 24


def check_time(iteration, start, end):
    return start <= iteration % 24 < end


def new_day(iteration):
    return iteration % 24 == 0


def day_of_week(iteration):
    return (iteration // 24) % 7 + 1


def work_day(iteration):
    wd = day_of_week(iteration)
    return wd not in [1, 7]


def day_of_month(iteration):
    return (iteration // 24) % 30 + 1


def new_month(iteration):
    return day_of_month(iteration) == 1 and iteration % 24 == 0


def bed_time(iteration):
    return check_time(iteration,0,8)


def work_time(iteration):
    return check_time(iteration,8,16)


def lunch_time(iteration):
    return iteration % 24 == 12


def free_time(iteration):
    return check_time(iteration,17,24)