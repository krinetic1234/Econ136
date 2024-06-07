import random
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import copy
import statistics
import matplotlib.pyplot as plt

# Base class for Agents
class Agent:
    def __init__(self, name, current_location):
        self.name = name
        self.current_location = current_location

# Derived class for Drivers, inheriting from Agent
class Driver(Agent):
    def __init__(self, name, current_location):
        super().__init__(name, current_location)
        self.total_income = 0
        self.total_rides = 0
        self.preference_ordering = []
        self.matched_passenger_name = None
        self.locations = []
    
    def cost_to_driver(self, passenger, city_center):
        mean = 1
        sd = 0.2
        cost_coef = random.gauss(mean, sd)
        bare_cost = (abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] - self.current_location[1]) + 0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + (passenger.dropoff_location[1] - city_center[1])**2))**2)
        return cost_coef * bare_cost

    def calculate_preference_ordering_driver(self, passengers, city_center):
        p = 0.1
        self.preference_ordering = sorted(passengers, key=lambda passenger: -(passenger.WTP - 2*((1-p)*(abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] - self.current_location[1])) + p*0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + (passenger.dropoff_location[1] - city_center[1])**2))**2)))

    def calculate_preference_ordering_driver_new(self, passengers, city_center):
        p = 0.9
        self.preference_ordering = sorted(passengers, key=lambda passenger: -(passenger.WTP - 2*((1-p)*(abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] - self.current_location[1])) + p*0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + (passenger.dropoff_location[1] - city_center[1])**2))**2)))

# Derived class for Passengers, inheriting from Agent
class Passenger(Agent):
    def __init__(self, name, current_location, dropoff_location):
        super().__init__(name, current_location)
        self.dropoff_location = dropoff_location
        self.WTP = self.calculate_WTP()
        self.preference_ordering = []
        self.matched_driver_name = None
    
    def calculate_WTP(self):
        mean = 1
        sd = 0.1
        wtp = random.gauss(mean, sd)
        euclidean_distance1 = math.sqrt((self.dropoff_location[0] - 0)**2 + (self.dropoff_location[1] - 0)**2)
        euclidean_distance2 = math.sqrt((self.current_location[0] - 0)**2 + (self.current_location[1] - 0)**2)
        manhattan_distance = abs(self.current_location[0] - self.dropoff_location[0]) + abs(self.current_location[1] - self.dropoff_location[1])
        return wtp*((euclidean_distance1+ euclidean_distance2)/2 + manhattan_distance)/3

    def calculate_preference_ordering_passenger(self, drivers, var):
        def man_dist(driver):
            return (abs(driver.current_location[0] - self.current_location[0]) + abs(driver.current_location[1] - self.current_location[1]))
        
        driver_dict = {driver: (man_dist(driver), driver.total_income) for driver in drivers}
        values = list(driver_dict.values())
        values_function1, values_function2 = zip(*values)
        scaler = MinMaxScaler()
        normalized_function1 = scaler.fit_transform([[value] for value in values_function1])
        normalized_function2 = scaler.fit_transform([[value] for value in values_function2])
        normal_driver_dict = {driver: (normalized_function1[i][0] + var*normalized_function2[i][0]) for i, driver in enumerate(driver_dict.keys())}
        self.preference_ordering = sorted(normal_driver_dict.keys(), key=lambda key: normal_driver_dict[key])

def manh_dist(driver, passenger):
    return (abs(driver.current_location[0] - passenger.current_location[0]) + abs(driver.current_location[1] - passenger.current_location[1]))

def generate_drivers(num_drivers):
    drivers = []
    for i in range(1, num_drivers + 1):
        current_location = generate_coordinate_point()
        drivers.append(Driver(f"Driver_{i}", current_location))
    return drivers

def generate_passengers(num_passengers):
    passengers = []
    for i in range(1, num_passengers + 1):
        current_location = generate_coordinate_point()
        dropoff_location = generate_coordinate_point()
        passengers.append(Passenger(f"Passenger_{i}", current_location, dropoff_location))
    return passengers

def generate_coordinate_point():
    mean = 0
    std_deviation = 20
    x = int(random.gauss(mean, std_deviation))
    y = int(random.gauss(mean, std_deviation))
    x = max(min(x, 50), -50)
    y = max(min(y, 50), -50)
    return x, y

def random_matches(drivers, passengers):
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    random.shuffle(drivers)
    random.shuffle(passengers)
    matches = []
    for i, passenger in enumerate(passengers):
        driver = drivers[i % len(drivers)]
        if driver not in matches:
            matches.append((driver.name, passenger.name))
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)
    return matches

def closest_matches(drivers, passengers):
    driver_dict_name = {driver.name: driver for driver in drivers}
    passenger_dict_name = {passenger.name: passenger for passenger in passengers}
    driver_locations = [driver.current_location for driver in drivers]
    passenger_locations = [passenger.current_location for passenger in passengers]
    driver_dict = {driver.current_location: driver for driver in drivers}
    passenger_dict = {passenger.current_location: passenger for passenger in passengers}
    distance_matrix = cdist(driver_locations, passenger_locations)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matches = []
    for i, j in zip(row_ind, col_ind):
        driver_point = driver_locations[i]
        passenger_point = passenger_locations[j]
        driver = driver_dict[driver_point]
        passenger = passenger_dict[passenger_point]
        matches.append((driver.name, passenger.name))
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict_name[passenger_name].WTP < manh_dist(driver_dict_name[driver_name], passenger_dict_name[passenger_name]):
            matches.remove(match)
    return matches

def driver_proposing_matching(drivers, passengers, city_center, var):
    matches = []
    unmatched_passengers = passengers.copy()
    unmatched_drivers = drivers.copy()
    for i in range(len(drivers)):
        drivers[i].matched_passenger_name = None 
        drivers[i].calculate_preference_ordering_driver(passengers, city_center)
        passengers[i].matched_passenger_name = None
        passengers[i].calculate_preference_ordering_passenger(drivers, var)
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    passenger_prefs = {passenger.name: {driver.name: i for (i, driver) in enumerate(passenger.preference_ordering)} for passenger in passengers}
    driver_prefs = {driver.name: {passenger.name: i for (i, passenger) in enumerate(driver.preference_ordering)} for driver in drivers}
    free_drivers = set(drivers)
    while free_drivers:
        driver = free_drivers.pop()
        for preferred_passenger in driver.preference_ordering:
            if preferred_passenger.matched_driver_name is None:
                preferred_passenger.matched_driver_name = driver.name
                driver.matched_passenger_name = preferred_passenger.name
                break 
            else:
                current_match_rank = passenger_prefs[preferred_passenger.name][preferred_passenger.matched_driver_name]
                new_match_rank = passenger_prefs[preferred_passenger.name][driver.name]
                if new_match_rank < current_match_rank:
                    free_drivers.add(driver_dict[preferred_passenger.matched_driver_name])
                    preferred_passenger.matched_driver_name = driver.name
                    driver.matched_passenger_name = preferred_passenger.name
                    break
    matches = [(driver.name, driver.matched_passenger_name) for driver in drivers]
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)
    return matches

def driver_proposing_matching_new(drivers, passengers, city_center, var):
    matches = []
    unmatched_passengers = passengers.copy()
    unmatched_drivers = drivers.copy()
    for i in range(len(drivers)):
        drivers[i].matched_passenger_name = None 
        drivers[i].calculate_preference_ordering_driver_new(passengers, city_center)
        passengers[i].matched_passenger_name = None
        passengers[i].calculate_preference_ordering_passenger(drivers, var)
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    passenger_prefs = {passenger.name: {driver.name: i for (i, driver) in enumerate(passenger.preference_ordering)} for passenger in passengers}
    driver_prefs = {driver.name: {passenger.name: i for (i, passenger) in enumerate(driver.preference_ordering)} for driver in drivers}
    free_drivers = set(drivers)
    while free_drivers:
        driver = free_drivers.pop()
        for preferred_passenger in driver.preference_ordering:
            if preferred_passenger.matched_driver_name is None:
                preferred_passenger.matched_driver_name = driver.name
                driver.matched_passenger_name = preferred_passenger.name
                break 
            else:
                current_match_rank = passenger_prefs[preferred_passenger.name][preferred_passenger.matched_driver_name]
                new_match_rank = passenger_prefs[preferred_passenger.name][driver.name]
                if new_match_rank < current_match_rank:
                    free_drivers.add(driver_dict[preferred_passenger.matched_driver_name])
                    preferred_passenger.matched_driver_name = driver.name
                    driver.matched_passenger_name = preferred_passenger.name
                    break
    matches = [(driver.name, driver.matched_passenger_name) for driver in drivers]
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)
    return matches

def simulate_round(drivers, new_passengers, round_num, city_center, algo, var):
    print(f"Simulating round {round_num} with algorithm {algo.__name__}")
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in new_passengers}
    if algo == driver_proposing_matching:
        new_matches = driver_proposing_matching(drivers, new_passengers, city_center, 0)
    if algo == driver_proposing_matching_new:
        new_matches = driver_proposing_matching_new(drivers, new_passengers, city_center, 5)
    elif algo == random_matches:
        new_matches = random_matches(drivers, new_passengers)
    else:
        new_matches = closest_matches(drivers, new_passengers)
    for match in new_matches:
        for driver in drivers:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
        if driver.matched_passenger_name is not None:
            passenger = passenger_dict[driver.matched_passenger_name]
            driver.current_location = passenger.dropoff_location
            driver.total_income += passenger.WTP
    return new_matches, len(new_matches)

def simulations(NUM_AGENTS):
    round_num = 1
    city_center = (0, 0)
    drivers1 = generate_drivers(NUM_AGENTS)
    drivers2 = copy.deepcopy(drivers1)
    drivers3 = copy.deepcopy(drivers1)
    drivers4 = copy.deepcopy(drivers1)
    passengers = generate_passengers(NUM_AGENTS)
    passengers2 = copy.deepcopy(passengers)
    print("Initial matching")
    matches1 = driver_proposing_matching(drivers1, passengers, city_center, 0)
    matches2 = random_matches(drivers2, passengers)
    matches3 = closest_matches(drivers3, passengers)
    matches4 = driver_proposing_matching_new(drivers4, passengers2, city_center, 5)
    driver_dict = {driver.name: driver for driver in drivers1}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    passenger_dict_2 = {passenger.name: passenger for passenger in passengers2}
    for match in matches1:
        for driver in drivers1:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP
    for match in matches2:
        for driver in drivers2:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP
    for match in matches3:
        for driver in drivers3:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP
    for match in matches4:
        for driver in drivers4:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict_2[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP  
    total_len_1 = len(matches1)
    total_len_2 = len(matches2)  
    total_len_3 = len(matches3)  
    total_len_4 = len(matches4)            
    for round_num in range(2, 50):
        new_passengers = generate_passengers(NUM_AGENTS)
        new_passengers2 = copy.deepcopy(new_passengers)
        matches1, length1 = simulate_round(drivers1, new_passengers, round_num, city_center, driver_proposing_matching, 0)
        matches2, length2 = simulate_round(drivers2, new_passengers, round_num, city_center, random_matches, 0)
        matches3, length3 = simulate_round(drivers3, new_passengers, round_num, city_center, closest_matches, 0)
        matches4, length4 = simulate_round(drivers4, new_passengers2, round_num, city_center, driver_proposing_matching_new, 5)
        total_len_1 += length1
        total_len_2 += length2
        total_len_3 += length3
        total_len_4 += length4
    drivers_list = [drivers1, drivers2, drivers3, drivers4]
    length_list = [total_len_1, total_len_2, total_len_3, total_len_4]
    results = {}
    for i in range(4):
        total_incomes = [driver.total_income for driver in drivers_list[i]]
        revenue = sum(total_incomes)
        revenue_per_ride = revenue/length_list[i]
        mean_income = statistics.mean(total_incomes)
        sd_income = statistics.stdev(total_incomes)
        if i == 0:
            results["da"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        elif i == 1:
            results["random"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        elif i == 2:
            results["close"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        else:
            results["new_da"] = [revenue, revenue_per_ride, mean_income, sd_income] 
    return results

def main():
    revenue_da = []
    revenue_per_ride_da = []
    incomes_da = []
    sd_income_da = []
    revenue_new_da = []
    revenue_per_ride_new_da = []
    incomes_new_da = []
    sd_income_new_da = []
    
    for i in range(5, 50):
        print(f"Running simulations for {i} agents")
        results = simulations(i)
        revenue_da.append(results["da"][0])
        revenue_per_ride_da.append(results["da"][1])
        incomes_da.append(results["da"][2])
        sd_income_da.append(results["da"][3])
        revenue_new_da.append(results["new_da"][0])
        revenue_per_ride_new_da.append(results["new_da"][1])
        incomes_new_da.append(results["new_da"][2])
        sd_income_new_da.append(results["new_da"][3])

    x_values = list(range(5, 50))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(x_values, revenue_da, label='Old DA')
    axs[0, 0].plot(x_values, revenue_new_da, label='New DA')
    axs[0, 0].set_title('Total Revenue')
    axs[0, 0].set_xlabel('Number of Agents')
    axs[0, 0].set_ylabel('Revenue')
    axs[0, 0].legend()

    axs[0, 1].plot(x_values, revenue_per_ride_da, label='Old DA')
    axs[0, 1].plot(x_values, revenue_per_ride_new_da, label='New DA')
    axs[0, 1].set_title('Revenue per Ride')
    axs[0, 1].set_xlabel('Number of Agents')
    axs[0, 1].set_ylabel('Revenue per Ride')
    axs[0, 1].legend()

    axs[1, 0].plot(x_values, incomes_da, label='Old DA')
    axs[1, 0].plot(x_values, incomes_new_da, label='New DA')
    axs[1, 0].set_title('Mean Income per Driver')
    axs[1, 0].set_xlabel('Number of Agents')
    axs[1, 0].set_ylabel('Income per Driver')
    axs[1, 0].legend()

    axs[1, 1].plot(x_values, sd_income_da, label='Old DA')
    axs[1, 1].plot(x_values, sd_income_new_da, label='New DA')
    axs[1, 1].set_title('Standard Deviation of Income')
    axs[1, 1].set_xlabel('Number of Agents')
    axs[1, 1].set_ylabel('SD of Income')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
