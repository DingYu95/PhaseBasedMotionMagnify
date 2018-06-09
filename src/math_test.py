from math import sin, cos

M_PI = 3.1416
order = 16
fh = 0.7
fl = 0.4
for i in range(order):
    w = sin(2*M_PI*fh*(i-order/2))/(M_PI*(i-order/2)) - sin(2*M_PI*fl*(i-order/2))/(M_PI*(i-order/2))
    hw = 0.54 - 0.46 * cos(2*M_PI*i / order)
    print(w * hw)
