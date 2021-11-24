import random


class Uniform(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return random.uniform(self.a, self.b)


class Gaussian(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return random.gauss(self.mean, self.std)


class Constant(object):
    def __init__(self, val):
        self.val = val

    def sample(self):
        return self.val