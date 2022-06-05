import numpy as np
from time import time

class Sphere():
    def __init__(self, dimensions, max_value=100, min_value=-100):
        self.dim = dimensions
        self.max_environment = max_value
        self.min_environment = min_value
        
    def evaluate(self, positions):
        return np.sum(positions**2)

class Wolf():
    def __init__(self, gid, position):
        self.id = gid
        self.position = position
        self.fitness = np.inf

    def move(self, a, alpha, beta, delta):
        dim = len(self.position)

        # alpha
        r1 = np.random.uniform(size=dim)
        r2 = np.random.uniform(size=dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha.position - self.position)
        X1 = alpha.position - A1 * D_alpha

        #beta
        r3 = np.random.uniform(size=dim)
        r4 = np.random.uniform(size=dim)
        A2 = 2 * a * r3 - a
        C2 = 2 * r4
        D_beta = abs(C2 * beta.position - self.position)
        X2 = beta.position - A2 * D_beta

        #delta
        r5 = np.random.uniform(size=dim)
        r6 = np.random.uniform(size=dim)
        A3 = 2 * a * r5 - a
        C3 = 2 * r6
        D_delta = abs(C3 * delta.position - self.position)
        X3 = delta.position - A3 * D_delta

        self.position = (X1 + X2 + X3)/3


class GWO():
    def __init__(self, nagents, max_iter, dimensions, fitness_function, simulation_id):
        np.random.seed(simulation_id+int(time())) # more generic seed
        self.dimensions = dimensions
        self.fitness_function = fitness_function(dimensions=self.dimensions)
        self.nagents = nagents
        self.max_iter = max_iter
        self.pop = []
        self.alpha = None
        self.beta = None
        self.delta = None
        self.simulation_id = simulation_id
        self.pattern_name = f'GWO_simulation_{self.simulation_id}_'
        print(self.pattern_name)
        self.best_fitness_through_iterations = []

    def _initialize(self):
        self.pop.clear()
        self.alpha = Wolf(-1, []) #dumb wolves
        self.beta = Wolf(-2, [])
        self.delta = Wolf(-3, [])
        for i in range(self.nagents):
            position = np.random.uniform(self.fitness_function.min_environment, self.fitness_function.max_environment, self.dimensions)
            wolf = Wolf(i, position)
            wolf.fitness = self.fitness_function.evaluate(wolf.position)
            self.pop.append(wolf)
        self._update_leaders()

    def _update_leaders(self):
        cloned_pop = self.pop[:]
        np.random.shuffle(cloned_pop)
        ranked = sorted(cloned_pop, key = lambda wolf: wolf.fitness)
        self.alpha = ranked[0]
        self.beta = ranked[1]
        self.delta = ranked[2]

    def optimize(self, debug=False):
        i = 1
        self._initialize()
        self.best_fitness_through_iterations = []
        while i <= self.max_iter:
            a = 2 - i * (2 / self.max_iter)

            for wolf in self.pop:
                wolf.move(a, self.alpha, self.beta, self.delta)
                new_fit = self.fitness_function.evaluate(wolf.position)
                wolf.fitness = new_fit
            self._update_leaders()

            if debug and i % 100 == 0:
                print("Simu: %d  -   Iteration: %d    -    Best Fitness: %e   -   best id %d" % (self.simulation_id, i, self.alpha.fitness, self.alpha.id))

            self.best_fitness_through_iterations.append(self.alpha.fitness)
            i+=1

        np.savetxt(self.pattern_name + 'best_fitness_through_iterations.txt',
                   self.best_fitness_through_iterations, fmt='%.4e')

        return self.alpha.position, self.best_fitness_through_iterations

if __name__ == '__main__':
    dimensions = 100
    simulations = 10
    max_iterations = 1000
    agents = 30
    func = Sphere
    print(f"Running {simulations} simulations using {func.__class__.__name__} function")
    bfs = []
    for i in range(simulations):
        gwo = GWO(agents, max_iterations, dimensions, func, i)
        best_fitness, convergence = gwo.optimize(True)
        bfs.append(best_fitness)
        print(f"#{i:02d}: {best_fitness}")
    print(f"\nMean: {np.mean(bfs)} | +- {np.std(bfs)}")