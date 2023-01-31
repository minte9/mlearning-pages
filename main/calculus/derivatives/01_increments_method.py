""" Increments method

An average speed of 30 mi/hr does not necessary means an  
exact speed for 3 hours. In the case of a ball droped near the 
surface of the earth, the formula for distance traveled is: 
    s = 16t^2 ft/sec

The instant speed is not the quontient of distance and time,
it is the limit of average speeds at exactly t=4.
Using increments method, the numbers seem getting closer to:
    128 ft/sec
"""

# Average speed during the fifth second approximation
s = 16*5**2 - 16*4**2
print(s) 
    # 144

# Improve the approximation using interval from 4 to 4.1 seconds
s = (16*4.1**2 - 16*4**2) / 0.1
print(round(s, 1)) 
    # 129.6

# The method of increments
def speed(t1, rate=1):
    t2 = t1 + rate
    s = 16 * (t2**2 - t1**2) / rate
    s = round(s, 1)
    print(f'{t1} to {t2} seconds: va = {s} ft/sec')

speed(4, 1)         # 144.0
speed(4, 0.1)       # 129.6
speed(4, 0.01)      # 128.2
speed(4, 0.001)     # 128.0
speed(4, 0.0001)    # 128.0
speed(4, 0.00001)   # 128.0