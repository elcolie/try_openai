import typing as typ

import numpy as np
from scipy import interpolate

# x
red_press = [
    130,
    672,
    1145,
    1615,
    2774,
]
# y
red_tvd = [
    0,
    4647.69236938775,
    5979.87795306123,
    6647.96503469388,
    9440.24,
]


def second_order_derivative(x_list, y_list) -> typ.List[typ.Dict]:
    """
    Return 2nd order derivative.

    f''(x) = (f(x+h) - 2*f(x) _ f(x-h))/h**2
    f(x) = x_list[i]
    f(x+h) = x_list[i+1]
    f(x-h) = x_list[i-1]
    Therefore edge of the x_list will have no 2nd derivative value. x_list[0], x_list[-1]
    """
    # Do linear interpolation first
    _f = interpolate.interp1d(x_list, y_list)
    xnew = np.linspace(x_list[0], x_list[-1], len(x_list))
    ynew = _f(xnew)
    delta_x = xnew[1] - xnew[0]
    ans: typ.List[typ.Dict] = []
    for i, (x_new, y_new) in enumerate(zip(xnew, ynew)):
        if i not in [0, len(xnew) - 1]:
            acc = (ynew[i - 1] - 2 * ynew[i] + ynew[i - 1]) / (delta_x)
            _ans = {
                "index": i,
                "value": acc
            }
            ans.append(_ans)
    return ans


mm = second_order_derivative(red_press, red_tvd)
print(mm)
