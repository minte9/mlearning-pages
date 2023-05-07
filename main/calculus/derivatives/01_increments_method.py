""" Increments method

An average speed of 30 mi/hr does not necessary means an  
exact speed for 3 hours. 

In the case of a ball droped near the surface of the earth, 
the formula for distance traveled is: 
    s = 16t^2 ft/sec

The instant speed is not the quontient of distance and time,
it is the limit of average speeds at exactly t=4.

Using increments method, the numbers seem getting closer to:
    s4 = 128 ft/sec
"""

# Average speed aproximation
s1 = 16*5**2 - 16*4**2               # from 4 to 5 seconds
s2 = (16*4.1**2 - 16*4**2) / 0.1    # from 4 to 4.1 seconds
s3 = (16*4.1**2 - 16*4**2) / 0.01  # from 4 to 4.01 seconds

# -------------------------------------------------------------------

def speed(t1, rate=1):
    t2 = t1 + rate
    s = 16 * (t2**2 - t1**2) / rate # The method of increments
    s = round(s, 1)
    return (t1, t2, s)

# -------------------------------------------------------------------

print("Manual aproximation:")
print("4 to 5s: speed =", s1)
print("4 to 4.1s: speed =", s2)
print("4 to 4.01s: speed = ", s3, "\n")

print("Increment method speed() function:")
print('%s to %ss: %s ft/sec' % speed(4, 1))
print('%s to %ss: %s ft/sec' % speed(4, 0.1))
print('%s to %ss: %s ft/sec' % speed(4, 0.01))
print('%s to %ss: %s ft/sec' % speed(4, 0.001))
print('%s to %ss: %s ft/sec' % speed(4, 0.0001))
print('%s to %ss: %s ft/sec' % speed(4, 0.00001))

"""
    Manual aproximation:
     4 to 5s: speed = 144
     4 to 4.1s: speed = 129.5999999999998
     4 to 4.01s: speed =  1295.999999999998 

    Increment method speed() function:
     4 to 5s: 144.0 ft/sec
     4 to 4.1s: 129.6 ft/sec
     4 to 4.01s: 128.2 ft/sec
     4 to 4.001s: 128.0 ft/sec
     4 to 4.0001s: 128.0 ft/sec
     4 to 4.00001s: 128.0 ft/sec
"""