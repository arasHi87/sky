def func_builder(name):
    def f():
        return name
    return f