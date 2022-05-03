import datetime


def return_first_path_for_date(products, dt: datetime.datetime):
    filtered_products = [path for (date_time, path) in products if dt == date_time]
    if len(filtered_products) < 1:
        raise ValueError(f'Could not find product for date {dt.isoformat()}')

    return filtered_products[0]


def return_product_paths_for_dt(l1c_products, l2a_products, acolite_products, dt: datetime.datetime):
    l1c_path = return_first_path_for_date(l1c_products, dt)
    l2a_path = return_first_path_for_date(l2a_products, dt)
    acolite_path = return_first_path_for_date(acolite_products, dt)

    return l1c_path, l2a_path, acolite_path
