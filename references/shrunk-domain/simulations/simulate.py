import time

from simulation import Simulation
from simulation_options import SimulationOptions


def main():
    opt = SimulationOptions().parse()
    start = time.time()
    sim = Simulation(opt)
    sim.simulate_save()
    end = time.time()
    time_sim = end - start
    print("elapsed time {:.0f}[min] {:.1f}[s]".format(time_sim // 60, time_sim % 60))


if __name__ == "__main__":
    main()
