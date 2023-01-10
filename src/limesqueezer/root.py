from .auxiliaries import wait, G, Function, Any
#%%═══════════════════════════════════════════════════════════════════════════
## ROOT FINDING
def interval(f: Function, n1: int, y1: float, n2: int, y2: float, fit1
             ) -> tuple[int, Any]:
    '''Returns the last x where f(x)<0'''
    while n2 - n1 > 2:
        # Arithmetic mean between linear estimate and half
        x_mid = int((n1 - y1 / (y2 - y1) * (n2 - n1) + (n2 + n1) / 2) / 2)
        if x_mid == n1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == n2:
            x_mid -= 1

        y_mid, fit2 = f(x_mid)

        if y_mid > 0:
            n2, y2 = x_mid, y_mid
        else:
            n1, y1, fit1 = x_mid, y_mid, fit2
    if n2 - n1 == 2: # Points have only one point in between
        y_mid, fit2 = f(n1+1) # Testing that point
        return (n1+1, fit2) if (y_mid <0) else (n1, fit1) # If under, give that fit
    else:
        return n1, fit1
#───────────────────────────────────────────────────────────────────────
def interval_debug(f: Function, n1: int, y1: float, n2: int, y2: float, fit1
                   ) -> tuple[int, Any]:
    '''Returns the last x where f(x)<0'''

    print(f'\t{n1=}\t{y1=}')
    print(f'\t{n2=}\t{y2=}')
    G['mid'], = G['ax_root'].plot(n1, y1,'.', color = 'blue')

    while n2 - n1 > 2:
        # Arithmetic mean between linear estimate and half
        linest = n1 - y1 / (y2 - y1) * (n2 - n1)
        halfest = (n2 + n1) / 2
        # sqrtest1 = sqrtx1 - y1 * (sqrtx2 - sqrtx1) / (y2 - y1)
        # sqrtest1 = sqrtest1*sqrtest1
        # sqrtest2 = int(n1 + (n2 - n1) / (y2 / y1 - 1)**2)
        # x_mid = int((n2 + n1) / 2)
        x_mid = int((linest + halfest) / 2)
        if x_mid == n1:    # To stop repetition in close cases
            x_mid += 1
        elif x_mid == n2:
            x_mid -= 1

        y_mid, fit2 = f(x_mid)

        print(f'\t{x_mid=}\t{y_mid=}')
        G['mid'].set_xdata(x_mid)
        G['mid'].set_ydata(y_mid)

        if y_mid > 0:
            wait('\tError over tolerance\n')
            G['ax_root'].plot(n2, y2,'.', color = 'black')

            n2, y2 = x_mid, y_mid

            G['xy2'].set_xdata(n2)
            G['xy2'].set_ydata(y2)
        else:
            wait('\tError under tolerance\n')
            G['ax_root'].plot(n1, y1,'.', color = 'black')
            n1, y1, fit1 = x_mid, y_mid, fit2
            G['xy1'].set_xdata(n1)
            G['xy1'].set_ydata(y1)
    if n2 - n1 == 2: # Points have only one point in between
        y_mid, fit2 = f(n1+1) # Testing that point
        return (n1+1, fit2) if (y_mid <0) else (n1, fit1) # If under, give that fit
    else:
        return n1, fit1
#───────────────────────────────────────────────────────────────────────
def droot(f: Function, y1: float, n2: int, limit: int) -> tuple[int, Any]:
    '''Finds the upper limit to an interval'''
    n1 = 0
    y2, fit2 = f(n2)
    fit1 = None

    while y2 < 0:

        n1, y1, fit1 = n2, y2, fit2
        n2 *= 2
        n2 += 1

        if n2 >= limit:
            y2, fit2 = f(limit)
            if y2 < 0:
                return limit, fit2
            else:
                n2 = limit
                break
        y2, fit2 = f(n2)
    return interval(f, n1, y1, n2, y2, fit1)
#───────────────────────────────────────────────────────────────────────
def droot_debug(f: Function, y1: float, n2: int, limit: int) -> tuple[int, Any]:
    '''Finds the upper limit to an interval
    '''
    n1 = 0
    y2, fit2 = f(n2)
    fit1 = None

    G['xy1'], = G['ax_root'].plot(n1, y1,'g.')
    G['xy2'], = G['ax_root'].plot(n2, y2,'b.')
    G['ax_root'].plot(n1, y1,'k.')
    while y2 < 0:
        wait('Calculating new attempt in droot\n')

        n1, y1, fit1 = n2, y2, fit2
        n2 *= 2
        n2 += 1

        print(f'{limit=}')
        print(f'{n1=}\t{y1=}')
        print(f'{n2=}\t{y2=}')
        G['xy1'].set_xdata(n1)
        G['xy1'].set_ydata(y1)
        G['xy2'].set_xdata(n2)

        if n2 >= limit:
            G['ax_root'].plot([limit, limit], [y1,0],'b.')
            y2, fit2 = f(limit)
            if y2<0:
                wait('End reached within tolerance\n')
                return limit, fit2
            else:
                wait('End reached outside tolerance\n')
                n2 = limit
                break
        y2, fit2 = f(n2)

        G['ax_root'].plot(n1, y1,'k.')
        print(f'{n1=}\t{y1=}')
        print(f'{n2=}\t{y2=}')
        G['ax_root'].plot(n2, y2,'k.')
        G['xy2'].set_ydata(y2)

    G['xy2'].set_color('red')
    wait('Points for interval found\n')
    return interval_debug(f, n1, y1, n2, y2, fit1)
#%%═══════════════════════════════════════════════════════════════════════════
_intervals = (interval, interval_debug)
_droots = (droot, droot_debug)
__all__ = ['_intervals' '_droots']