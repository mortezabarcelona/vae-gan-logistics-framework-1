import pandas as pd
import numpy as np
import os
import time as timemodule
from geopy.distance import geodesic
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure 'data/synthetic' directory exists
os.makedirs("data/synthetic", exist_ok=True)

# Parameters
num_samples = 10000
scenario = 'baseline'
total_hours = 168

# European logistics hubs
cities = [
    'London', 'Paris', 'Berlin', 'Amsterdam', 'Madrid', 'Rome', 'Warsaw', 'Vienna', 'Prague', 'Budapest',
    'Brussels', 'Barcelona', 'Milan', 'Hamburg', 'Munich', 'Frankfurt', 'Cologne', 'Lyon', 'Marseille', 'Copenhagen',
    'Rotterdam', 'Antwerp', 'Valencia', 'Genoa'
]

# Seaports (expanded list)
seaports = {'Amsterdam', 'Barcelona', 'Hamburg', 'Marseille', 'Copenhagen', 'Rotterdam', 'Antwerp', 'Valencia', 'Genoa'}

# City coordinates (added new cities)
city_coords = {
    'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522), 'Berlin': (52.52, 13.405), 'Amsterdam': (52.3676, 4.9041),
    'Madrid': (40.4168, -3.7038), 'Rome': (41.9028, 12.4964), 'Warsaw': (52.2297, 21.0122),
    'Vienna': (48.2082, 16.3738), 'Prague': (50.0755, 14.4378), 'Budapest': (47.4979, 19.0402),
    'Brussels': (50.8503, 4.3517), 'Barcelona': (41.3851, 2.1734), 'Milan': (45.4642, 9.1900),
    'Hamburg': (53.5511, 9.9937), 'Munich': (48.1351, 11.5820), 'Frankfurt': (50.1109, 8.6821),
    'Cologne': (50.9375, 6.9603), 'Lyon': (45.7640, 4.8357), 'Marseille': (43.2965, 5.3698),
    'Copenhagen': (55.6761, 12.5683), 'Rotterdam': (51.9244, 4.4777), 'Antwerp': (51.2213, 4.4051),
    'Valencia': (39.4699, -0.3763), 'Genoa': (44.4056, 8.9463)
}

# Precompute distances
logger.info("Precomputing city distances")
city_distances = {(c1, c2): int(geodesic(city_coords[c1], city_coords[c2]).km) for c1 in cities for c2 in cities if
                  c1 != c2}

# Initialize dataset
logger.info("Initializing dataset")
shipment_ids = range(num_samples)
time_steps = np.random.randint(0, total_hours, num_samples)
days = [(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])[(t // 24) % 7] for t in
        time_steps]
hours = [t % 24 for t in time_steps]

data = {
    'shipment_id': shipment_ids,
    'time_step': time_steps,
    'day_of_week': days,
    'hour_of_day': hours
}

# Scenario configuration
if scenario == 'baseline':
    demand_range = (0.8, 1.2)
    port_issue_prob = 0.05
    emission_penalty_rate = 0.15
    weather_probs = {'clear': 0.7, 'rain': 0.2, 'storm': 0.05, 'fog': 0.05}

# Shipment features
logger.info("Generating shipment features")
demand = np.random.uniform(*demand_range, num_samples)
volumes = np.random.normal(50, 10, num_samples) * demand
weights = np.random.normal(1000, 200, num_samples) * demand

# Routing with forced seaport routes
logger.info("Generating routes")
min_sea_routes = int(0.25 * num_samples)  # Ensure at least 25% sea routes
sea_routes_assigned = 0
origins = np.random.choice(cities, num_samples)
destinations = np.random.choice(cities, num_samples)
for i in range(num_samples):
    while origins[i] == destinations[i]:
        destinations[i] = np.random.choice(cities)
    # Force some seaport-to-seaport routes
    if sea_routes_assigned < min_sea_routes and np.random.rand() < 0.5 and origins[i] in seaports and destinations[
        i] in seaports:
        sea_routes_assigned += 1


def generate_route(origin, destination):
    route = [(origin, destination)]
    if np.random.rand() < 0.5:
        mid = np.random.choice([c for c in cities if c not in {origin, destination}])
        route = [(origin, mid), (mid, destination)]
    if np.random.rand() < 0.3:
        second_mid = np.random.choice([c for c in cities if c not in set(sum(route, ()))])
        route = [(origin, route[0][1]), (route[0][1], second_mid), (second_mid, destination)]
    return route


routes = [generate_route(origins[i], destinations[i]) for i in range(num_samples)]


# Assign transport modes
def assign_mode(city1, city2):
    dist = city_distances.get((city1, city2), city_distances.get((city2, city1)))
    if dist > 100 and city1 in seaports and city2 in seaports:
        return np.random.choice(['road', 'rail', 'sea', 'air'], p=[0.05, 0.05, 0.7, 0.2])
    elif dist > 500:
        return np.random.choice(['road', 'rail', 'air'], p=[0.3, 0.3, 0.4])
    return np.random.choice(['road', 'rail'], p=[0.4, 0.6])


modes = []
leg_distances = []
for r in routes:
    mode_list, dist_list = [], []
    for leg in r:
        mode = assign_mode(*leg)
        mode_list.append(mode)
        dist_list.append(city_distances.get(leg, city_distances.get(leg[::-1])))
    modes.append(mode_list)
    leg_distances.append(dist_list)

# Enforce minimum mode proportions through resampling
logger.info("Enforcing minimum transport mode proportions")
target_proportion = 0.25  # Target 25% for each mode
mode_counts = pd.Series([m[0] if len(m) > 0 else 'road' for m in modes]).value_counts()
total = len(modes)

# Prioritize sea mode resampling
required_samples = {mode: max(0, int(target_proportion * total) - mode_counts.get(mode, 0)) for mode in
                    ['sea', 'air', 'road', 'rail']}
for mode, additional in required_samples.items():
    if additional > 0:
        # Find eligible routes for the mode
        eligible_indices = []
        for i, r in enumerate(routes):
            city1, city2 = r[0][0], r[-1][-1]
            dist = city_distances.get((city1, city2), city_distances.get((city2, city1)))
            if mode == 'sea' and (dist > 100 and city1 in seaports and city2 in seaports):
                eligible_indices.append(i)
            elif mode == 'air' and dist > 500:
                eligible_indices.append(i)
            elif mode in ['road', 'rail'] and dist <= 1000:
                eligible_indices.append(i)

        # Resample to add the mode
        if eligible_indices:
            to_resample = np.random.choice(eligible_indices, size=min(additional, len(eligible_indices)), replace=False)
            for idx in to_resample:
                modes[idx] = [mode] * len(modes[idx])

# Cap overrepresented modes
mode_counts = pd.Series([m[0] if len(m) > 0 else 'road' for m in modes]).value_counts(normalize=True)
for mode in ['rail', 'road', 'air']:  # Process overrepresented modes first
    if mode_counts.get(mode, 0) > 0.25:
        excess = int((mode_counts[mode] - 0.25) * total)
        # Find routes using this mode
        indices_to_redistribute = [i for i, m in enumerate(modes) if m[0] == mode]
        if indices_to_redistribute:
            to_redistribute = np.random.choice(indices_to_redistribute, size=min(excess, len(indices_to_redistribute)),
                                               replace=False)
            # Redistribute to sea if eligible, otherwise to other modes
            for idx in to_redistribute:
                city1, city2 = routes[idx][0][0], routes[idx][-1][-1]
                dist = city_distances.get((city1, city2), city_distances.get((city2, city1)))
                # Prioritize sea
                if dist > 100 and city1 in seaports and city2 in seaports:
                    modes[idx] = ['sea'] * len(modes[idx])
                else:
                    other_modes = [m for m in ['road', 'rail', 'air', 'sea'] if
                                   m != mode and mode_counts.get(m, 0) < 0.25]
                    if other_modes:
                        new_mode = np.random.choice(other_modes)
                        if new_mode == 'sea' and not (dist > 100 and city1 in seaports and city2 in seaports):
                            continue
                        if new_mode == 'air' and dist <= 500:
                            continue
                        if new_mode in ['road', 'rail'] and dist > 1000:
                            continue
                        modes[idx] = [new_mode] * len(modes[idx])

# Adjust volume and weight for air mode
for i in range(num_samples):
    primary_mode = modes[i][0]
    if primary_mode == 'air':
        volumes[i] = min(volumes[i], 10)  # Max 10 m³ for air
        weights[i] = min(weights[i], 5000)  # Max 5000 kg for air

# Assign fields
data['demand_level'] = demand
data['volume'] = volumes
data['weight'] = weights
data['origin'] = origins
data['destination'] = destinations
data['distance'] = [sum(dists) for dists in leg_distances]
data['transport_mode'] = [m[0] if len(m) > 0 else 'road' for m in modes]
data['route'] = [' -> '.join([f"{start}-{end}({mode})" for (start, end), mode in zip(r, m)]) for r, m in
                 zip(routes, modes)]

# Delivery deadlines (hours since time_step)
min_times = np.array(data['distance']) / 30 + 10
max_times = min_times + 240
data['delivery_deadline'] = [ts + np.random.uniform(mn, mx) for ts, mn, mx in
                             zip(data['time_step'], min_times, max_times)]

# Weather, traffic, port status, incidents, fatigue
logger.info("Generating disruption features")
data['weather_condition'] = np.random.choice(list(weather_probs.keys()), num_samples, p=list(weather_probs.values()))
data['weather_severity'] = [np.random.uniform(0, 1) if w != 'clear' else 0 for w in data['weather_condition']]

traffic = []
for i in range(num_samples):
    h = data['hour_of_day'][i]
    d = data['day_of_week'][i]
    w = data['weather_condition'][i]
    base = np.random.uniform(1.5, 2.5) if (7 <= h <= 9 or 17 <= h <= 19) and d in ['Monday', 'Tuesday', 'Wednesday',
                                                                                   'Thursday',
                                                                                   'Friday'] else np.random.uniform(1.0,
                                                                                                                    1.5)
    traffic_value = base * (1 + data['weather_severity'][i]) if w in ['storm', 'fog'] else base
    traffic.append(np.clip(traffic_value, 0, 3))  # Normalize to [0, 3]
data['traffic_congestion'] = traffic

# Port status and road incidents
port_stat = []
incidents = []
fatigue = []
driver_counts = {}
for i in range(num_samples):
    mode = data['transport_mode'][i]
    if mode == 'sea':
        port_stat.append(np.random.choice(['open', 'delayed', 'closed'], p=[0.95, 0.03, 0.02]))
    else:
        port_stat.append('open')
    if mode == 'road':
        incidents.append(np.random.choice(['none', 'accident', 'construction'], p=[0.95, 0.025, 0.025]))
        driver_id = f"driver_{i % 100}"
        driver_counts[driver_id] = driver_counts.get(driver_id, 0) + 1
        fatigue.append(min(1.0, driver_counts[driver_id] * 0.15))
    else:
        incidents.append('none')
        fatigue.append(0.0)
data['port_status'] = port_stat
data['road_incident'] = incidents
data['driver_fatigue'] = fatigue

# Port congestion
port_entries = [(data['time_step'][i], data['destination'][i], data['volume'][i]) for i in range(num_samples) if
                data['transport_mode'][i] == 'sea']
port_load = \
pd.DataFrame(port_entries, columns=['time_step', 'destination', 'volume']).groupby(['time_step', 'destination'])[
    'volume'].sum().to_dict()
port_cap = {c: 5000 for c in cities}
data['port_congestion'] = [
    min(1.0, port_load.get((data['time_step'][i], data['destination'][i]), 0) / port_cap[data['destination'][i]]) if
    data['transport_mode'][i] == 'sea' else 0.0 for i in range(num_samples)]

# Transit time, emissions, cost
logger.info("Calculating transit times, emissions, and costs")
speeds = {'road': 60, 'rail': 80, 'sea': 30, 'air': 600}  # km/h
co2_factors = {'road': 0.1, 'rail': 0.03, 'sea': 0.01, 'air': 0.5}  # g CO2/ton-km
cost_factors = {'road': 0.5, 'rail': 0.3, 'sea': 0.2, 'air': 1.0}  # cost/km

transit = []
fuel_price = []
co2 = []
fuel_cost = []
penalties = []
for i in range(num_samples):
    m = data['transport_mode'][i]
    d = data['distance'][i]
    # Base transit time
    t = d / speeds[m]
    if m != 'air':
        t *= data['traffic_congestion'][i] * (1 + data['weather_severity'][i])
        if m == 'sea':
            if data['port_status'][i] == 'delayed': t *= 1.5
            if data['port_status'][i] == 'closed': t *= 2.0
            t *= (1 + data['port_congestion'][i])
        elif m == 'road':
            if data['road_incident'][i] != 'none': t *= 1.3
            t *= (1 + data['driver_fatigue'][i])
    else:
        t = max(2, t)  # Minimum 2 hours for air (e.g., airport processing)
    t = np.clip(t, 0, 240)  # Cap at 10 days
    transit.append(t)

    base_price = cost_factors[m] * (1 + 0.2 * np.sin(2 * np.pi * data['time_step'][i] / total_hours))
    jitter = np.random.normal(0, 0.05 * base_price)
    price = max(0.05, base_price + jitter)
    fuel_price.append(price)
    fuel_cost.append(price * d)

    e = co2_factors[m] * d * data['weight'][i] / 1000  # Convert weight to tons
    if m != 'air':
        e *= (1 + 0.1 * data['traffic_congestion'][i]) * (1 + 0.05 * data['driver_fatigue'][i])
    e += np.random.normal(0, 0.02 * e)
    co2.append(max(0, e))

    urgency_factor = 1 + 0.3 * np.random.rand()
    p = emission_penalty_rate * e * urgency_factor if m in ['air', 'road'] else emission_penalty_rate * e
    penalties.append(p)

data['transit_time'] = transit
data['fuel_price'] = fuel_price
data['co2_emissions'] = co2
data['fuel_cost'] = fuel_cost
data['emission_penalty'] = penalties

# Urgency & satisfaction
logger.info("Generating urgency and satisfaction")
urgency = []
satisfaction = []
for i in range(num_samples):
    prio = {'low': 0.3, 'medium': 0.6, 'high': 0.9}[np.random.choice(['low', 'medium', 'high'])]
    pressure = 1 - (data['delivery_deadline'][i] - data['time_step'][i]) / 300
    u = max(0, min(1, 0.5 * prio + 0.5 * pressure))
    urgency.append(u)
    deadline_diff = max(1, data['delivery_deadline'][i] - data['time_step'][i])
    sat = max(0, min(1, 1 - (data['transit_time'][i] / deadline_diff) * u))
    satisfaction.append(np.random.beta(2, 5) * sat)  # Increase variance
data['shipment_urgency'] = urgency
data['customer_satisfaction'] = satisfaction

# Validate mode distribution
logger.info("Validating transport mode distribution")
mode_counts = pd.Series(data['transport_mode']).value_counts(normalize=True)
logger.info(f"Transport mode distribution:\n{mode_counts}")
threshold = 0.15  # Minimum proportion for each mode
imbalanced = mode_counts[mode_counts < threshold]
if not imbalanced.empty:
    logger.warning(f"Imbalanced transport modes (below {threshold * 100}%):\n{imbalanced}")

# Save dataset to synthetic directory
timestamp = int(timemodule.time())
output_path = f"data/synthetic/logistics_data_{scenario}_{timestamp}.csv"
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
logger.info(f"Synthetic dataset for {scenario} with {num_samples} entries saved to {output_path}")