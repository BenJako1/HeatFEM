
def unpack_dict(input_dict, ):
    # Keys and their lists
    keys = list(input_dict.keys())
    values = [input_dict[k] for k in keys]

    # Ensure all lists are same length
    n = len(values[0])
    if not all(len(v) == n for v in values):
        raise ValueError("All dictionary value lists must have the same length")

    # Prepare a dict of unpacked lists for each key
    unpacked = {k: [] for k in keys}

    # Iterate over all entries in parallel
    for i in range(n):
        entry_items = {k: input_dict[k][i] for k in keys}

        # Detect if any value is a range string (like '3:7' or '3:10:2')
        # and use its length as the expansion length
        main_len = None
        if any(isinstance(v, str) and ":" in v for v in entry_items.values()):
            # Pick the first such range to determine the expansion count
            for v in entry_items.values():
                if isinstance(v, str) and ":" in v:
                    parts = [int(x) for x in v.split(":")]
                    range_vals = list(range(*parts))
                    main_len = len(range_vals)
                    break
        elif any(isinstance(v, (list, tuple)) for v in entry_items.values()):
            for v in entry_items.values():
                if isinstance(v, (list, tuple)):
                    main_len = len(v)
                    break
        else:
            main_len = 1

        # Now expand all keys based on that length
        for k, v in entry_items.items():
            if isinstance(v, str) and ":" in v:
                parts = [int(x) for x in v.split(":")]
                unpacked[k].extend(range(*parts))
            elif isinstance(v, (list, tuple)):
                unpacked[k].extend(v)
            else:
                unpacked[k].extend([v] * main_len)

    return unpacked