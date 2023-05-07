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

# Average speed for 4 to 5 s approximation
s45 = 16*5**2 - 16*4**2

# Average speed from 4 to 4.1 s approximation
s45_1 = (16*4.1**2 - 16*4**2) / 0.1

# The method of increments
def speed(t1, rate=1):
    t2 = t1 + rate
    s = 16 * (t2**2 - t1**2) / rate
    s = round(s, 1)
    return (t1, t2, s)


print("Average speed for 4 to 5 second approximation: \n", s45)
print("Average speed from 4 to 4.1 seconds approximation: \n", s45_1)

print('%s to %s seconds: speed = %s ft/sec' % speed(4, 1))
print('%s to %s seconds: speed = %s ft/sec' % speed(4, 0.1))
print('%s to %s seconds: speed = %s ft/sec' % speed(4, 0.01))
print('%s to %s seconds: speed = %s ft/sec' % speed(4, 0.001))
print('%s to %s seconds: speed = %s ft/sec' % speed(4, 0.0001))
print('%s to %s seconds: speed = %s ft/sec' % speed(4, 0.00001))

"""
    Average speed for 4 to 5 second approximation: 
     144
    Average speed from 4 to 4.1 seconds approximation: 
     129.5999999999998
    
    4 to 5 seconds: va = 144.0 ft/sec
    4 to 4.1 seconds: va = 129.6 ft/sec
    4 to 4.01 seconds: va = 128.2 ft/sec
    4 to 4.001 seconds: va = 128.0 ft/sec
    4 to 4.0001 seconds: va = 128.0 ft/sec
    4 to 4.00001 seconds: va = 128.0 ft/sec
"""