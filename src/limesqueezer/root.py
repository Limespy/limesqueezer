from . import GLOBALS
from .auxiliaries import wait
global G
G = GLOBALS.dictionary
#%%═════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f, x1, y1, x2, y2, fit1):
    '''Returns the last x where f(x)<0'''
    while x2 - x1 > 2:
        # Arithmetic mean between linear estimate and half
        x_mid = int((x1 - y1 / (y2 - y1) * (x2 - x1) + (x2 + x1) / 2) / 2)
        if x_mid == x1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == x2:
            x_mid -= 1

        y_mid, fit2 = f(x_mid)

        if y_mid > 0:
            x2, y2 = x_mid, y_mid
        else:
            x1, y1, fit1 = x_mid, y_mid, fit2
    if x2 - x1 == 2: # Points have only one point in between
        y_mid, fit2 = f(x1+1) # Testing that point
        return (x1+1, fit2) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def interval_debug(f, x1, y1, x2, y2, fit1):
    '''Returns the last x where f(x)<0'''

    print(f'\t{x1=}\t{y1=}')
    print(f'\t{x2=}\t{y2=}')
    G['mid'], = G['ax_root'].plot(x1, y1,'.', color = 'blue')

    while x2 - x1 > 2:
        # Arithmetic mean between linear estimate and half
        linest = x1 - y1 / (y2 - y1) * (x2 - x1)
        halfest = (x2 + x1) / 2
        # sqrtest1 = sqrtx1 - y1 * (sqrtx2 - sqrtx1) / (y2 - y1)
        # sqrtest1 = sqrtest1*sqrtest1
        # sqrtest2 = int(x1 + (x2 - x1) / (y2 / y1 - 1)**2)
        # x_mid = int((x2 + x1) / 2)
        x_mid = int((linest + halfest) / 2)
        if x_mid == x1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == x2:
            x_mid -= 1

        y_mid, fit2 = f(x_mid)

        print(f'\t{x_mid=}\t{y_mid=}')
        G['mid'].set_xdata(x_mid)
        G['mid'].set_ydata(y_mid)

        if y_mid > 0:
            wait('\tError over tolerance\n')
            G['ax_root'].plot(x2, y2,'.', color = 'black')

            x2, y2 = x_mid, y_mid

            G['xy2'].set_xdata(x2)
            G['xy2'].set_ydata(y2)
        else:
            wait('\tError under tolerance\n')
            G['ax_root'].plot(x1, y1,'.', color = 'black')
            x1, y1, fit1 = x_mid, y_mid, fit2
            G['xy1'].set_xdata(x1)
            G['xy1'].set_ydata(y1)
    if x2 - x1 == 2: # Points have only one point in between
        y_mid, fit2 = f(x1+1) # Testing that point
        return (x1+1, fit2) if (y_mid <0) else (x1, fit1) # If under, give that fit
    else:
        return x1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''

    x1 = 0
    y2, fit2 = f(x2)
    fit1 = None

    while y2 < 0:

        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1

        if x2 >= limit:
            y2, fit2 = f(limit)
            if y2 < 0:
                return limit, fit2
            else:
                x2 = limit
                break
        y2, fit2 = f(x2)
    return interval(f, x1, y1, x2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
def droot_debug(f, y1, x2, limit):
    '''Finds the upper limit to interval
    '''
    x1 = 0
    y2, fit2 = f(x2)
    fit1 = None

    G['xy1'], = G['ax_root'].plot(x1, y1,'g.')
    G['xy2'], = G['ax_root'].plot(x2, y2,'b.')

    while y2 < 0:

        wait('Calculating new attempt in droot\n')
        G['ax_root'].plot(x1, y1,'k.')

        x1, y1, fit1 = x2, y2, fit2
        x2 *= 2
        x2 += 1

        print(f'{limit=}')
        print(f'{x1=}\t{y1=}')
        print(f'{x2=}\t{y2=}')
        G['xy1'].set_xdata(x1)
        G['xy1'].set_ydata(y1)
        G['xy2'].set_xdata(x2)

        if x2 >= limit:
            G['ax_root'].plot([limit, limit], [y1,0],'b.')
            y2, fit2 = f(limit)
            if y2<0:
                wait('End reached within tolerance\n')
                return limit, fit2
            else:
                wait('End reached outside tolerance\n')
                x2 = limit
                break
        y2, fit2 = f(x2)

        G['ax_root'].plot(x1, y1,'k.')
        print(f'{x1=}\t{y1=}')
        print(f'{x2=}\t{y2=}')
        G['ax_root'].plot(x2, y2,'k.')
        G['xy2'].set_ydata(y2)

    G['xy2'].set_color('red')
    wait('Points for interval found\n')
    return interval_debug(f, x1, y1, x2, y2, fit1)
