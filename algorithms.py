
def summation(n, term):
    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total

def cube(x):
    return x * x * x

def sum_cubes(n):
    return summation(n, cube)

def identity(x):
    return x

def sum_naturals(n):
    return summation(n, identity)

def pi_term(x):
    return 8 / ((4 * x - 3) * (4 * x - 1))

def pi_sum(n):
    return summation(n, pi_term)



def improve(update, close, guess = 1, max_update=100):
    k = 0
    while not close(guess) and k < max_update:
        guess = update(guess)
        print(guess)
        k = k + 1
    return guess

def golden_update(guess):
    return 1/guess + 1

def square_close_to_successor(guess):
    return approx_eq(1 / guess, guess - 1) 
    "return approx_eq(guess * guess, guess + 1)"

def approx_eq(x, y, tolerance=1e-16):
    return abs(x - y) < tolerance

def golden_ratio():
    def update(guess):
        return 1/guess + 1
    def close(guess):
        return approx_eq(1 / guess, guess - 1, 1e-16)
    return improve(update, close)




def average(x, y):
    return (x + y)/2

def sqrt_update_can_do_just_once(x, a):
    return average(x, a/x)

def sqrt(a):
    def sqrt_update(x):
        return average(x, a/x)
    def sqrt_close(x):
        return approx_eq(x * x, a)
    return improve(sqrt_update, sqrt_close)

def improve_test():
    phi = 1/2 + sqrt(5)/2
    approx_phi = improve(golden_update, square_close_to_successor)
    assert approx_eq(phi, approx_phi), 'phi differs from its approximation'


   
def cubrt(a):
    def cubrt_update(x, a):
        return (2*x + a/(x*x))/3
    return improve(lambda x: cubrt_update(x, a),
            lambda x: approx_eq(x*x*x, a))

def find_zero(f, df):
    def near_zero(x):
        return approx_eq(f(x), 0)
    return improve(newton_update(f, df), near_zero)

def newton_update(f, df):
    def update(x):
        return x - f(x) / df(x)
    return update

def square_root(a):
    def f(x):
        return x*x - a
    def df(x):
        return 2*x
    return find_zero(f, df)

def cube_root(a):
    return find_zero(lambda x: x*x*x-a,
            lambda x: 3*x*x)


def power(x, n):
    """
    Returns x^n
    """
    product, k = 1, 0
    while k < n:
        product, k = product * x, k + 1
    return product

def root(n, a):
    def f(x):
        return power(x, n) - a
    def df(x):
        return n * power(x, n-1)
    return find_zero(f, df)


def make_deriv(f):
    def df(n):
        return f(n+1)  - f(n)
    return df


def is_prime(n):
    """
    >>> is_prime(10)
    False
    >>> is_prime(7)
    True
    """

    m = n - 1

    while m > 1:
        if n % m == 0:
            return False
        m = m - 1

    return True


def is_prime_efficient(n):
    """
    >>> is_prime(10)
    False
    >>> is_prime(7)
    True
    """

    if n == 1 or n == 2:
        return True

    """m = sqrt_int_plus_one(n) /// 
       m = int(sqrt(n)) + 1 ///Older square root methods """
    m = int(square_root(n)) + 1

    while m > 1:
        if n % m == 0:
            return False
        m = m - 1

    return True

def sqrt_rnd_up(n):
    """
    >>> sqrt_rnd_up(25)
    5
    >>> sqrt_rnd_up(13)
    4
    >>> sqrt_rnd_up(1)
    1
    """

    m = 1
    while m * m < n:
        m = m + 1

    return m

def sqrt_int_plus_one(n):
    n **= (1/2)
    return int(n) + 1



def make_adder(n):
    """Return a function that takes a number K and returns K + N

    >>> add_three = make_adder(3)
    >>> add_three(4)
    7
    """
    def adder(k):
        return k + n
    return adder



def square(x):
    return x * x

def successor(x):
    return x + 1

def compose1(f, g):
    def h(x):
        return f(g(x))
    return h

square_of_successor = compose1(square, successor)



def fib(n):
    """Compute the nth Fibonacci number. Note that the single-lined assignment cannotacheive same result if rewritten into two lines. Needs a temp to do so. See fib_with_temp.

    >>> fib(0)
    0
    >>> fib(8)
    21
    """
    k, num, last = 0, 0, 1
    while k < n:
        num, last = num + last, num
        k = k + 1
    return num




def fib_with_temp(n):
    """Compute the nth Fibonacci number.

    >>> fib(0)
    0
    >>> fib(8)
    21
    """
    k, num, last = 0, 0, 1
    while k < n:
        temp = num
        num = num + last
        last = temp
        k = k + 1
    return num


def sequence(term, n):
    """Returns the values of a sequence in order.

    >>> sequence(fib, 7)
    0
    1
    1
    2
    3
    5
    8
    13
    """

    k = 0
    while k <= n:
        print(term(k))
        term(k)
        k
        k = k + 1
    return "Figure this out later; use tuples?"


def k_amount(k):
    """Turns out k_amount cannot return multiple values"""
    while k >= 0:
        return k
        k = k - 1





def map_to_range(start, end, f):
    while start <= end:
        print(f(start))
        start = start + 1

def curry2(f):
    """Return a curried version of the given two-argument function."""
    def g(x):
        def h(y):
            return f(x, y)
        return h
    return g

def uncurry2(g):
    """Return a two-argument version of the given curried function."""
    def f(x, y):
        return g(x)(y)
    return f


def trace(fn):
    """Returns a version of fn that prints before it is
    called. The functions below use trace as a decorator."""
    def wrapped(x):
        print('-> ', fn, '(', x, ')')
        return fn(x)
    return wrapped

"""below are decorated functions. In general, these are equiv to assigning 
fn = higher_order_fn(fn) after def statement"""

@trace
def triple(x):
    return 3 * x

@trace
def sum_squares_up_to(n):
    k = 1
    total = 0
    while k <= n:
        total, k = total + square(k), k + 1
    return total



ab_plus_cd = lambda a,b,c,d: a*b + c*d

(lambda a,b,c,d,e,f: a*b + c*d + e*f)(1,2,3,4,5,6)


def gcd(a, b):
    if a == 0:
        return b
    return gcd(b%a, a)

def add_rationals(x, y):
    nx, dx = numer(x), denom(x)
    ny, dy = numer(y), denom(y)
    return rational(nx * dy + ny * dx, dx * dy)

def mul_rationals(x, y):
    return rational(numer(x) * numer(y), denom(x) * denom(y))

def div_rationals(x, y):
    return rational(numer(x) * denom(y), denom(x) * numer(y))

def square_rational(x):
    return mul_rationals(x, x)

def power_rational(x, n):
    """
    Returns x^n where x is a rational
    """
    product = rational(1, 1)
    k = 0
    while k < n:
        product, k = mul_rationals(product, x), k + 1
    return product


def print_rational(x):
    print(numer(x), '/', denom(x))

def rationals_are_equal(x, y):
    return (numer(x) * denom(y) == numer(y) * denom(x))


def pair(x, y):
    """Return a function that represents a pair."""
    def get(index):
        if index == 0:
            return x
        elif index == 1:
            return y
        else:
            return "Invalid index!"
    return get

def select(p, i):
    """Return the element at index i of pair p."""
    return p(i)

def rational(n, d):
    """Returns rational reduced to lowest terms"""
    g = gcd(n, d)
    return pair(n//g, d//g)

def numer(x):
    return select(x, 0)

def denom(x):
    return select(x, 1)









def sum_digits(n):
    """Return the sum of the digits of positive integer n."""
    if n < 10:
        return n
    else:
        all_but_last, last = n // 10, n % 10
        return sum_digits(all_but_last) + last

def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)

def fact_iter(n):
    total, k = 1, 1
    while k <= n:
        total, k = total*k, k+1
    return total

def split(n):
    """
    Note: Typerror if split(split(n)) is called
    """
    return n // 10, n % 10

def luhn_sum(n):
    if n < 10:
        return n
    else:
        all_but_last, last = split(n)
        return luhn_sum_double(all_but_last) + last

def luhn_sum_double(n):
    all_but_last, last = split(n)
    luhn_digit = sum_digits(2 * last)
    if n < 10:
        return luhn_digit
    else:
        return luhn_sum(all_but_last) + luhn_digit

def is_luhn_valid(n):
    return luhn_sum(n) % 10 == 0

def how_many_luhn_valid(n):
    """
    Calculates how many numbers n are luhn valid.
    Results: 9 if n=1e2
            99 if n=1e3
           999 if n=ne4
          9999 if n=ne5 
    You see the pattern!
    So the average 16-digit CC is one of 900 trillion valid numbers.
    """
    total = 0
    while n > 0:
        if is_luhn_valid(n):
            total = total + 1
        n = n - 1
    return total


def print_luhn_valid_nums(n):
    """
    Prints all luhn valid numbers to n inclusive.
    Be careful.
    """
    while n > 0:
        if is_luhn_valid(n):
            print(n)
        n = n - 1
    return print("End")


def is_even_mutual_rec(n):
    """Max recursion reached depth at n=998"""
    if n == 0:
        return True
    elif n > 0:
        return is_odd_mutual_rec(n-1)
    else:
        return is_odd_mutual_rec(n+1)


def is_odd_mutual_rec(n):
    """Max recursion reached depth at n=998"""
    if n == 0:
        return False
    elif n > 0:
        return is_even_mutual_rec(n-1)
    else:
        return is_even_mutual_rec(n+1)

def is_even_single_rec(n):
    """Max recursion depth reached at n=1996"""
    if n == 0:
        return True
    elif n > 0:
        if n-1 == 0:
            return False
        else:
            return is_even_single_rec((n-1)-1)
    else:
        if n+1 == 0:
            return False
        else:
            return is_even_single_rec((n+1)+1)

def is_odd_single_rec(n):
    """Max recursion depth reached at n=1996"""
    if n == 0:
        return False
    elif n > 0:
        if n-1 == 0:
            return True
        else:
            return is_odd_single_rec((n-1)-1)
    else:
        if n+1 == 0:
            return True
        else:
            return is_odd_single_rec((n+1)+1)


def is_even_single_rec(n):
    """Max recursion depth reached at n=1996"""
    if n == 0:
        return True
    elif n > 0:
        if n-1 == 0:
            return False
        else:
            return is_even_single_rec((n-1)-1)
    else:
        if n+1 == 0:
            return False
        else:
            return is_even_single_rec((n+1)+1)


def is_even_iter(n):
    """
    Takes ~7 secs at n=1e8
    Takes ~55 secs at n=1e9
    """
    if n == 0:
        return True
    elif n > 0:
        k = n
        while k > 0:
            k = k - 2
            if k == 0:
                return True
        return False
    else:
        k = n
        while k < 0:
            k = k + 2
            if k == 0:
                return True
        return False


def is_odd_iter(n):
    """
    Takes ~7 secs at n=1e8
    Takes ~55 secs at n=1e9
    """
    if n == 0:
        return False
    elif n > 0:
        k = n
        while k > 0:
            k = k - 2
            if k == 0:
                return False
        return True
    else:
        k = n
        while k < 0:
            k = k + 2
            if k == 0:
                return False
        return True


def is_even(n):
    if n == 0:
        return True
    elif n == 1:
        return False
    else:
        return is_even(n % 2)

def is_odd(n):
    if n == 0:
        return False
    elif n == 1:
        return True
    else:
        return is_odd(n % 2)

def cascade(n):
    if n < 10:
        print(n)
    else:
        print(n)
        cascade(n//10)
        print(n)

def inverse_cascade(n):
    inverse_cascade_first_half(n)
    inverse_cascade_last_half(n//10)

def inverse_cascade_first_half(n):
    if n < 10:
        print(n)
    else:
        inverse_cascade_first_half(n//10)
        print(n)

def inverse_cascade_last_half(n):
    if n >= 10:
        print(n)
        inverse_cascade_last_half(n//10)
    else:
        print(n)

def inverse_cascade_alt(n):
    grow(n)
    print(n)
    shrink(n)

def f_then_g(f, g, n):
    if n:
        f(n)
        g(n)

grow = lambda n: f_then_g(grow, print, n//10)
shrink = lambda n: f_then_g(print, shrink, n//10)


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2) + fib(n-1)



def trace_update(fn):
    """Returns a version of fn that prints before it is
    called. The functions below use trace as a decorator."""
    space_length = [0]

    def wrapped(x):
        space = ' ' * space_length[0]
        space_length[0] = space_length[0] + 1
        print(space, fn.__name__, '(', x, ')', ':', sep='')
        result = fn(x)
        print(space, fn.__name__, '(', x, ')', ' -> ', result, sep='')
        space_length[0] = space_length[0] - 1
        return result
    return wrapped



@trace_update
def fib_traced(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_traced(n-2) + fib_traced(n-1)








