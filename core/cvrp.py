"""
CVRP Problem Definition Module
Defines the CVRP problem and its components
"""

import numpy as np
import math
import random
import json


class Customer:
    """Class representing a customer in the CVRP problem"""
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand


class CVRP:
    """Capacitated Vehicle Routing Problem class"""
    def __init__(self, capacity=100):
        """Initialize a CVRP problem"""
        self.customers = []  # List of customers, index 0 is depot
        self.depot = None    # The depot
        self.capacity = capacity  # Vehicle capacity
        self.distances = None  # Distance matrix

    def add_depot(self, x, y):
        """Add a depot to the problem"""
        self.depot = Customer(0, x, y, 0)
        self.customers.append(self.depot)

    def add_customer(self, id, x, y, demand):
        """Add a customer to the problem"""
        customer = Customer(id, x, y, demand)
        self.customers.append(customer)

    def load_problem(self, num_customers, capacity, seed=None):
        """Create a random CVRP problem"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.capacity = capacity
        self.customers = []

        # Place depot at (0, 0)
        self.add_depot(0, 0)

        # Create random customers
        for i in range(1, num_customers + 1):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            demand = random.randint(10, 40)  # Random demand
            self.add_customer(i, x, y, demand)

        # Calculate distance matrix
        self.calculate_distances()

    def load_from_file(self, filename):
        """Load CVRP problem from a file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

                self.capacity = data['capacity']
                self.customers = []

                depot = data['depot']
                self.add_depot(depot['x'], depot['y'])

                for c in data['customers']:
                    self.add_customer(c['id'], c['x'], c['y'], c['demand'])

                self.calculate_distances()
                return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

    def save_to_file(self, filename):
        """Save CVRP problem to a file"""
        try:
            data = {
                'capacity': self.capacity,
                'depot': {
                    'x': self.depot.x,
                    'y': self.depot.y
                },
                'customers': [
                    {
                        'id': c.id,
                        'x': c.x,
                        'y': c.y,
                        'demand': c.demand
                    } for c in self.customers[1:]  # Skip depot
                ]
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False

    def calculate_distances(self):
        """Calculate distance matrix between customers"""
        n = len(self.customers)
        self.distances = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    xi, yi = self.customers[i].x, self.customers[i].y
                    xj, yj = self.customers[j].x, self.customers[j].y
                    self.distances[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def calculate_route_distance(self, route):
        """Calculate the distance of a route"""
        if not route:
            return 0

        distance = 0
        prev_node = 0  # Start from depot

        for node in route:
            distance += self.distances[prev_node][node]
            prev_node = node

        # Return to depot
        distance += self.distances[prev_node][0]

        return distance

    def calculate_route_demand(self, route):
        """Calculate the total demand of a route"""
        return sum(self.customers[i].demand for i in route)

    def calculate_solution_cost(self, solution):
        """Calculate the total distance of a solution (list of routes)"""
        total_distance = sum(self.calculate_route_distance(route) for route in solution)
        return total_distance

    def is_solution_valid(self, solution):
        """Check if a solution is valid"""
        # Check if each route exceeds capacity
        for route in solution:
            if self.calculate_route_demand(route) > self.capacity:
                return False

        # Check if each customer is served exactly once
        visited = set()
        for route in solution:
            for customer in route:
                if customer in visited:
                    return False
                visited.add(customer)

        # Check if all customers are served
        all_customers = set(range(1, len(self.customers)))
        if visited != all_customers:
            return False

        return True

    def get_unvisited_customers(self, visited):
        """Get the list of unvisited customers"""
        all_customers = set(range(1, len(self.customers)))
        return list(all_customers - set(visited))

    def save_to_json(self, filename):
        """Save CVRP problem to a json file"""
        return self.save_to_file(filename)
        
    def load_from_json(self, filename):
        """Load CVRP problem from a json file"""
        return self.load_from_file(filename)