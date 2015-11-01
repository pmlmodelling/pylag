import numpy as np

def round_time(dt, round_to=3600):
    """
    Round given datetime object to the number of given seconds
    
    Parameters:
    -----------
    dt: Datetime
        Date and time
        
    round_to: int
        No. of seconds to round to (default 3600, or one hour)
        
    Returns:
    --------
    -/-: Datetime
        Rounded datetime object
    """
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to/2) // round_to * round_to
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def sort_adjacency_array(nv, nbe):
    """
    Sort nbe array for FlowVC. This step is critical, as the algorithm FlowVC
    uses to identify neighbouring elements a particle has moved into depends on
    the correct sorting of nbe.
    
    Parameters:
    -----------
    nv: 2D ndarray, int
        Nodes surrounding element with shape (3, n_elems)
        
    nbe: 2D ndarray, int
        Elements surrounding element with shape (3, n_elems)
        
    Returns:
    --------
    nbe_sorted: 2D ndarray, int
        The new nbe array, sorted for FlowVC.
    """
    n_elems = nv.shape[1]

    # Our new to-be-sorted nbe array
    nbe_sorted = np.zeros([3,n_elems], dtype='int') - 1

    # Loop over all elems
    for i in range(n_elems):
        side1, side2, side3 = _get_empty_arrays()

        side1[0] = nv[2,i]
        side1[1] = nv[0,i]
        side2[0] = nv[0,i]
        side2[1] = nv[1,i]
        side3[0] = nv[1,i]
        side3[1] = nv[2,i]

        index_side1 = -1
        index_side2 = -1
        index_side3 = -1
        for j in range(3):
            elem = nbe[j,i]
            if elem != -1:
                nv_test = nv[:,elem]
                if _get_number_of_matching_nodes(nv_test, side1) == 2:
                    index_side1 = elem
                elif _get_number_of_matching_nodes(nv_test, side2) == 2:
                    index_side2 = elem
                elif _get_number_of_matching_nodes(nv_test, side3) == 2:
                    index_side3 = elem
                else:
                    raise Exception('Failed to match side to test element.')

        nbe_sorted[0,i] = index_side1
        nbe_sorted[1,i] = index_side2
        nbe_sorted[2,i] = index_side3

    return nbe_sorted
        
def _get_empty_arrays():
    side1 = np.empty(2)
    side2 = np.empty(2)
    side3 = np.empty(2)
    return side1, side2, side3

def _get_number_of_matching_nodes(array1, array2):
    match = 0
    for a1 in array1:
        for a2 in array2:
            if a1 == a2: match = match + 1

    return match

