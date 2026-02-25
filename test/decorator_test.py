def decorator1(func):
    def wrapper():
        print("Decorator 1")
        func()
    return wrapper

def decorator2(func):
    def wrapper():
        print("Decorator 2")
        func()
    return wrapper
def test(func):
    def wrapper():
        print("Test Decorator")
        func()
    return wrapper

@decorator1
@decorator2
@test
def say_hello():
    print("Hello!")

say_hello()

