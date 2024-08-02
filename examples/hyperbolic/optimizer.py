import numpy as np

class CEM:
    def __init__(self, mean, sigma_init, sigma_end, decay_iteration, population_size):
        self.population_size = population_size
        self._sigma_init = sigma_init
        self._sigma_end = sigma_end
        self._sigma = self._sigma_init
        self._mean = mean
        self.data_shape = mean.shape

        self.decay_iteration = decay_iteration
        self.update(0)
        self.solutions = []
        self.xs = []
        self.xs_best = []
    
    def update(self, iteration):
        self._sigma = max((self._sigma_end - self._sigma_init) / self.decay_iteration * iteration + self._sigma_init, 
                          self._sigma_end) 
        print(f"update sigma to: {self._sigma}")

    def ask(self):
        if len(self.xs) > 0:
            x =  self.xs.pop(-1) 
        elif len(self.xs_best) > 0: 
            z = np.random.randn(*self.data_shape)
            x = self.xs_best.pop(-1) + z * self._sigma
        else:
            z = np.random.randn(*self.data_shape)
            x = self._mean + z * self._sigma
        return x

    def tell(self, solutions):
        solutions.sort(key=lambda s: s[1])
        solutions = solutions[-int(self.population_size * 0.5):] 
        self.xs = [s[0] for s in solutions]
        self.xs_best = [s[0] for s in solutions]
        x_k = np.array([s[0] for s in solutions]) 
        self._mean = x_k.mean(axis=0)

    def should_stop(self):
        return False