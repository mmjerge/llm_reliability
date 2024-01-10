def pytest_runtest_protocol(item, nextitem):
    """Summary

    Parameters
    ----------
    item : _type_
        _description_
    nextitem : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Author
    ------
    Michael M. Jerge
    """
    description_marker = item.get_closest_marker("description")
    if description_marker is not None:
        print(f"\nDescription: {description_marker.args[0]}")
    author_marker = item.get_closest_marker("author")
    if author_marker is not None:
        print(f"Author: {author_marker.args[0]}")
    return None