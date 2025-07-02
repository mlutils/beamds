


from beam.utils import remote_debugger

remote_debugger()



# some random code
def some_function():
    a = 1
    b = 2
    c = a + b
    # add some prints
    print(f"a: {a}, b: {b}, c: {c}")
    # add some sleep
    time.sleep(1)

    return c

some_function()