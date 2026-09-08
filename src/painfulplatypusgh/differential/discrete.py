#///script
#requires-python=">=3.12"
#dependencies=[

#]
#///

def diff(t, x) :    # Takes two parameters: t (array of time values), x (signal value)
    
    if len(t) != len(x):    #Check for equal array length
        raise ValueError("Parameter arrays must be the same length.")

    v_t = []    #Initialize empty array
    for k in range(1, len(t)):  #Start at second index since the formula looks one index backwards
        v_k = (x[k]-x[k-1])/(t[k]-t[k-1])
        v_t.append(v_k) #Append derivative at certain index to array

    return v_t  #Return the array of derivatives