import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RideSharingPlatform:
    def __init__(self, drivers, riders):
        self.drivers = drivers
        self.riders = riders
        self.matches = []
        self.fairness_factors = {}  # To store fairness factors for pricing

    def two_stage_matching(self):
        # Stage 1: Immediate matching
        for rider in self.riders:
            if not self.drivers:
                break  # No drivers left to match
            closest_driver = self.find_closest_driver(rider)
            self.matches.append((rider, closest_driver))
            self.drivers.remove(closest_driver)
        
        # Stage 2: Future demand optimization
        future_demand = self.predict_future_demand()
        for demand in future_demand:
            if not self.drivers:
                break  # No drivers left to match
            closest_driver = self.find_closest_driver(demand)
            self.matches.append((demand, closest_driver))
            self.drivers.remove(closest_driver)
    
    def find_closest_driver(self, rider):
        distances = [self.calculate_distance(rider, driver) for driver in self.drivers]
        return self.drivers[np.argmin(distances)]

    def calculate_distance(self, rider, driver):
        return np.sqrt((rider['x'] - driver['x'])**2 + (rider['y'] - driver['y'])**2)

    def predict_future_demand(self):
        # Placeholder for future demand prediction logic
        # Simulate some future demands based on historical patterns
        future_riders = [{'id': 100 + i, 'x': np.random.rand(), 'y': np.random.rand()} for i in range(5)]
        return future_riders

    def route_splitting(self):
        # Implement route splitting logic
        for match in self.matches:
            rider, driver = match
            # Split the driver's route into two segments for illustration
            mid_point = {'x': (rider['x'] + driver['x']) / 2, 'y': (rider['y'] + driver['y']) / 2}
            segment1 = {'driver': driver, 'rider': rider, 'to': mid_point}
            segment2 = {'driver': driver, 'from': mid_point, 'to': rider}
            print(f"Route for driver {driver['id']} split into: {segment1} and {segment2}")

    def fairness_pricing(self):
        # Implement fairness-aware dynamic pricing logic
        for match in self.matches:
            rider, driver = match
            detour_distance = self.calculate_distance(rider, driver) * 0.1  # Simulate detour
            shared_ride_discount = 0.05  # Simulate shared ride discount factor
            fare = 10  # Base fare
            adjusted_fare = fare - (fare * detour_distance) - (fare * shared_ride_discount)
            match_id = (rider['id'], driver['id'])
            self.fairness_factors[match_id] = adjusted_fare
            print(f"Fairness pricing for match (Rider {rider['id']}, Driver {driver['id']}): {adjusted_fare}")

    def simulate(self):
        self.two_stage_matching()
        self.route_splitting()
        self.fairness_pricing()
        return self.matches, self.fairness_factors

# Sample data generation
def generate_sample_data(n_drivers, n_riders):
    drivers = [{'id': i, 'x': np.random.rand(), 'y': np.random.rand()} for i in range(n_drivers)]
    riders = [{'id': i, 'x': np.random.rand(), 'y': np.random.rand()} for i in range(n_riders)]
    return drivers, riders

# Algorithm comparison
def compare_algorithms():
    algorithms = ['Gale Shapley', 'Ratings']
    n_drivers, n_riders = 10, 10

    aggregate_income = []
    revenue_per_ride = []

    for algo in algorithms:
        drivers, riders = generate_sample_data(n_drivers, n_riders)
        if algo == 'Gale Shapley':
            platform = RideSharingPlatform(drivers, riders)
            matches, fairness_factors = platform.simulate()
        elif algo == 'Proximity':
            platform = ProximityMatchingPlatform(drivers, riders)
            matches, fairness_factors = platform.simulate()
        elif algo == 'Ratings':
            platform = RatingsMatchingPlatform(drivers, riders)
            matches, fairness_factors = platform.simulate()

        total_income = sum(fairness_factors.values())
        avg_revenue = total_income / len(matches) if matches else 0
        aggregate_income.append(total_income)
        revenue_per_ride.append(avg_revenue)

    generate_graphs(algorithms, aggregate_income, revenue_per_ride)

def generate_graphs(algorithms, aggregate_income, revenue_per_ride):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(algorithms, aggregate_income, color=['blue', 'green', 'red'])
    ax[0].set_title('Aggregate Income Comparison')
    ax[0].set_ylabel('Aggregate Income')
    ax[0].set_ylim(75, 100)  # Narrow y-axis limits for better visualization

    ax[1].bar(algorithms, revenue_per_ride, color=['blue', 'green', 'red'])
    ax[1].set_title('Revenue per Ride Comparison')
    ax[1].set_ylabel('Revenue per Ride')
    ax[1].set_ylim(7.5, 10)  # Narrow y-axis limits for better visualization

    plt.tight_layout()
    plt.show()

# Implementing Proximity and Ratings Matching Platforms
class ProximityMatchingPlatform(RideSharingPlatform):
    def two_stage_matching(self):
        # Single stage: Immediate matching based on proximity
        for rider in self.riders:
            if not self.drivers:
                break  # No drivers left to match
            closest_driver = self.find_closest_driver(rider)
            self.matches.append((rider, closest_driver))
            self.drivers.remove(closest_driver)

class RatingsMatchingPlatform(RideSharingPlatform):
    def two_stage_matching(self):
        # Single stage: Immediate matching based on ratings
        for rider in self.riders:
            if not self.drivers:
                break  # No drivers left to match
            highest_rated_driver = self.find_highest_rated_driver()
            self.matches.append((rider, highest_rated_driver))
            self.drivers.remove(highest_rated_driver)
    
    def find_highest_rated_driver(self):
        # Placeholder for finding the highest-rated driver
        # Assuming ratings are part of driver data
        return max(self.drivers, key=lambda driver: driver.get('rating', 0))

# Adding ratings to sample data
def generate_sample_data_with_ratings(n_drivers, n_riders):
    drivers = [{'id': i, 'x': np.random.rand(), 'y': np.random.rand(), 'rating': np.random.rand() * 5} for i in range(n_drivers)]
    riders = [{'id': i, 'x': np.random.rand(), 'y': np.random.rand()} for i in range(n_riders)]
    return drivers, riders

compare_algorithms()
