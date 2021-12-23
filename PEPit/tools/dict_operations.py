def merge_dict(dict1, dict2):
    """
    Merge keys of dict1 and dict2.
    If a key is in the 2 dictionaries, then add the values.

    Args:
        dict1 (dict): any dictionary
        dict2 (dict): any dictionary

    Returns:
        merged_dict (dict): the union of the 2 inputs with added values.

    """

    # Start from dict1
    merged_dict = dict1.copy()

    # Add all key of dict2 to dict1
    for key in dict2.keys():

        # If in both, the values are added
        if key in dict1.keys():

            merged_dict[key] += dict2[key]

        # Otherwise, just add the new key and value
        else:

            merged_dict[key] = dict2[key]

    # Return the merged dict
    return merged_dict


def prune_dict(my_dict):
    """
    Remove all keys associated to a null value.

    Args:
        my_dict (dict): any dictionary

    Returns:
        pruned_dict (dict): pruned dictionary

    """

    # Start from empty dict
    pruned_dict = dict()

    # Add all entry of dict1 that does not have a null value
    for key in my_dict.keys():

        if my_dict[key] != 0:

            pruned_dict[key] = my_dict[key]

    # Return pruned dict
    return pruned_dict


def multiply_dicts(dict1, dict2):
    """
    Multiply 2 dictionaries in the sense of developing a product of 2 sums.

    Args:
        dict1 (dict): any dictionary
        dict2 (dict): any dictionary

    Returns:
        product_dict (dict): the keys are the couple of keys of dict1 and dict2
                             and the values the product of values of dict1 and dict2.

    """

    # Start from empty dict
    product_dict = dict()

    # Complete the dict
    for key1 in dict1.keys():
        for key2 in dict2.keys():
            product_key = (key1, key2)
            product_value = dict1[key1] * dict2[key2]

            if product_key in product_dict.keys():
                product_dict[product_key] += product_value
            else:
                product_dict[product_key] = product_value

    # Return the product dict
    return product_dict
