import pandas as pd
import numpy as np
import pickle
import os
import random
from collections import defaultdict

def train_agent():
    # Paths
    pkl_in = "/Users/croma/Desktop/unsupervised_customer_segmentation/models/cleaned_food_data.pkl"
    models_dir = "/Users/croma/Desktop/unsupervised_customer_segmentation/models"
    
    if not os.path.exists(pkl_in):
        raise FileNotFoundError(f"Cleaned dataset not found at {pkl_in}. Run clean_data.py first.")
        
    print("Loading cleaned dataset...")
    with open(pkl_in, "rb") as f:
        df = pickle.load(f)
        
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Extract unique values
    countries = df['country_or_area'].unique()
    categories = df['category'].unique()
    
    # 4 Actions
    # 0: Standard Policy (sell normal price, normal storage, standard spoilage)
    # 1: Discount Price (-15%) (boosts demand, lower profit margin)
    # 2: Enhanced Refrigeration (halves spoilage, adds holding/energy cost)
    # 3: Menu/Recipe Repurposing (moderate demand boost, moderate cost & revenue)
    ACTIONS = [0, 1, 2, 3]
    num_actions = len(ACTIONS)
    
    # Hyperparameters
    alpha = 0.1         # Learning rate
    gamma = 0.9         # Discount factor
    epsilon = 1.0       # Initial exploration rate
    min_epsilon = 0.1
    epsilon_decay = 0.995
    episodes = 1000
    steps_per_episode = 30
    
    # Q-table: key is (country, category, stock_level, demand_level)
    # value is array of length 4 for Q-values of actions
    Q = defaultdict(lambda: np.zeros(num_actions))
    
    # Tracking statistics
    episode_rewards = []
    episode_steps = []
    episode_waste_reduction = []
    episode_profits = []
    
    # Base price and cost definitions
    # Map category to a base price scaling factor to make it realistic
    base_prices = {
        'cattle': 25.0,
        'chickens': 5.0,
        'goats': 15.0,
        'pigs': 18.0,
        'sheep': 16.0,
        'horses': 30.0,
        'asses': 12.0,
        'beehives': 8.0,
        'ducks': 6.0,
        'mules': 20.0
    }
    
    def get_base_price(cat):
        return base_prices.get(cat, 10.0)
        
    print("Training Q-learning agent...")
    
    for ep in range(episodes):
        # 1. Sample country and category to focus on for this episode
        country = random.choice(countries)
        category = random.choice(categories)
        
        # Determine average stock scale for this country-category to discretize stock
        sub_df = df[(df['country_or_area'] == country) & (df['category'] == category)]
        if len(sub_df) > 0:
            median_val = sub_df['value'].median()
        else:
            median_val = df[df['category'] == category]['value'].median()
        if pd.isna(median_val) or median_val == 0:
            median_val = 100000.0  # Fallback scale
            
        # Helper to map numerical stock value to discrete state
        def discretize_stock(val):
            if val < 0.5 * median_val:
                return 'Low'
            elif val < 1.5 * median_val:
                return 'Medium'
            else:
                return 'High'
                
        # Helper to convert discrete levels to index for transition calculations
        def level_to_val(lvl):
            if lvl == 'Low': return 1
            if lvl == 'Medium': return 2
            return 3
            
        def val_to_level(v):
            if v <= 1: return 'Low'
            if v == 2: return 'Medium'
            return 'High'

        # Initialize environment variables
        stock_val = random.choice([0.4, 1.0, 1.8]) * median_val
        stock_lvl = discretize_stock(stock_val)
        demand_lvl = random.choice(['Low', 'Medium', 'High'])
        
        state = (country, category, stock_lvl, demand_lvl)
        
        total_reward = 0
        total_steps = 0
        total_waste_reduction = 0
        total_profit = 0
        
        for step in range(steps_per_episode):
            total_steps += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = np.argmax(Q[state])
                
            # Simulate Environment Transition
            stock_v = level_to_val(stock_lvl)
            demand_v = level_to_val(demand_lvl)
            
            # Determine adjusted demand based on action
            if action == 1:   # Discount Price
                adj_demand = min(3, demand_v + 1)
            elif action == 3: # Menu Repurposing
                adj_demand = min(3.0, demand_v + 0.5)
            else:
                adj_demand = demand_v
                
            # Sales and stock dynamics
            sold = min(float(stock_v), float(adj_demand))
            remaining = max(0.0, float(stock_v) - sold)
            
            # Spoilage / waste rates
            # Action 2 (Refrigeration) reduces spoilage rate
            if action == 2:
                spoilage_rate = 0.1
            else:
                spoilage_rate = 0.4
                
            # Standard spoilage rate is 0.4. Calculate waste
            waste = remaining * spoilage_rate
            standard_waste = remaining * 0.4
            waste_reduction = max(0.0, standard_waste - waste)
            
            # Revenue calculation
            base_p = get_base_price(category)
            if action == 1:     # Discount (-15%)
                price = base_p * 0.85
            elif action == 3:   # Repurpose
                price = base_p * 0.90
            else:
                price = base_p
                
            revenue = sold * price
            
            # Cost calculation
            spoil_cost = waste * (base_p * 0.6)  # Spoilage is a waste loss
            
            if action == 2:     # Enhanced refrigeration energy cost
                holding_cost = base_p * 0.15
            else:
                holding_cost = 0.0
                
            if action == 3:     # Repurposing process cost
                repurpose_cost = base_p * 0.08
            else:
                repurpose_cost = 0.0
                
            total_cost = spoil_cost + holding_cost + repurpose_cost
            profit = revenue - total_cost
            
            # Reward formulation: emphasize profit and penalty on waste
            reward = profit - (waste * base_p * 0.2)
            
            # Determine next state
            # Next stock depends on remaining stock + replenishment
            replenishment = random.choice([0, 1, 2])
            next_stock_v = min(3, max(1, int(remaining) + replenishment))
            next_stock_lvl = val_to_level(next_stock_v)
            
            # Next demand is a random walk or random state
            next_demand_lvl = random.choice(['Low', 'Medium', 'High'])
            
            next_state = (country, category, next_stock_lvl, next_demand_lvl)
            
            # Update Q-table
            best_next_q = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next_q - Q[state][action])
            
            # Move to next state
            state = next_state
            stock_lvl = next_stock_lvl
            demand_lvl = next_demand_lvl
            
            # Accrue metrics
            total_reward += reward
            total_profit += profit
            total_waste_reduction += waste_reduction
            
        episode_rewards.append(total_reward)
        episode_steps.append(total_steps)
        episode_waste_reduction.append(total_waste_reduction)
        episode_profits.append(total_profit)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    print("Training finished.")
    
    # 1. Save Episodic Statistics
    stats_out = os.path.join(models_dir, "episode_stats.pkl")
    stats_data = {
        "rewards": episode_rewards,
        "steps": episode_steps,
        "waste_reduction": episode_waste_reduction,
        "profits": episode_profits
    }
    with open(stats_out, "wb") as f:
        pickle.dump(stats_data, f)
    print(f"Saved Episodic Statistics to {stats_out}")
    
    # 2. Save State-Action Mapping (Q-table)
    q_table_out = os.path.join(models_dir, "q_table.pkl")
    # Convert defaultdict to normal dict for clean pickling
    normal_q_table = {k: v.tolist() for k, v in Q.items()}
    with open(q_table_out, "wb") as f:
        pickle.dump(normal_q_table, f)
    print(f"Saved Q-Table to {q_table_out}")
    
    # 3. Save Action Probabilities (Policy)
    # Compute Softmax Policy from Q-values: P(a|s) = exp(Q(s,a)) / sum(exp(Q(s,b)))
    policy_out = os.path.join(models_dir, "policy.pkl")
    policy = {}
    for state_key, q_vals in Q.items():
        # Prevent overflow by subtracting max
        shifted_q = q_vals - np.max(q_vals)
        exp_q = np.exp(shifted_q)
        probs = exp_q / np.sum(exp_q)
        policy[state_key] = probs.tolist()
        
    with open(policy_out, "wb") as f:
        pickle.dump(policy, f)
    print(f"Saved Policy to {policy_out}")
    
    # 4. Save State Space Logs (Evolution of states in a simulation)
    # Run a final simulation of 50 steps using the learned greedy policy
    state_logs_out = os.path.join(models_dir, "state_logs.pkl")
    state_logs = []
    
    # Select a sample country and category for the log
    test_country = "China" if "China" in countries else countries[0]
    test_category = "cattle" if "cattle" in categories else categories[0]
    
    # Discretization baseline median
    sub_df = df[(df['country_or_area'] == test_country) & (df['category'] == test_category)]
    median_val = sub_df['value'].median() if len(sub_df) > 0 else df[df['category'] == test_category]['value'].median()
    if pd.isna(median_val) or median_val == 0:
        median_val = 100000.0
        
    # Start simulation
    stock_lvl = 'Medium'
    demand_lvl = 'Medium'
    
    print(f"Running simulation forecast for {test_country} - {test_category}...")
    for step in range(50):
        state = (test_country, test_category, stock_lvl, demand_lvl)
        # Greedy action selection from trained Q-table
        action = int(np.argmax(Q[state]))
        
        # System dynamics
        stock_v = level_to_val(stock_lvl)
        demand_v = level_to_val(demand_lvl)
        
        if action == 1:
            adj_demand = min(3, demand_v + 1)
        elif action == 3:
            adj_demand = min(3.0, demand_v + 0.5)
        else:
            adj_demand = demand_v
            
        sold = min(float(stock_v), float(adj_demand))
        remaining = max(0.0, float(stock_v) - sold)
        
        spoilage_rate = 0.1 if action == 2 else 0.4
        waste = remaining * spoilage_rate
        
        base_p = get_base_price(test_category)
        price = base_p * 0.85 if action == 1 else (base_p * 0.90 if action == 3 else base_p)
        revenue = sold * price
        
        spoil_cost = waste * (base_p * 0.6)
        holding_cost = base_p * 0.15 if action == 2 else 0.0
        repurpose_cost = base_p * 0.08 if action == 3 else 0.0
        profit = revenue - (spoil_cost + holding_cost + repurpose_cost)
        
        state_logs.append({
            "step": step,
            "country": test_country,
            "category": test_category,
            "stock_level": stock_lvl,
            "demand_level": demand_lvl,
            "action": action,
            "action_desc": ["Maintain Standard", "Discount Price (-15%)", "Refrigerate Surplus", "Repurpose Surplus"][action],
            "inventory_units": stock_v,
            "demand_units": demand_v,
            "sold_units": sold,
            "waste_units": waste,
            "profit": profit
        })
        
        # Next state
        replenishment = random.choice([0, 1, 2])
        next_stock_v = min(3, max(1, int(remaining) + replenishment))
        stock_lvl = val_to_level(next_stock_v)
        # Demand fluctuates slightly
        demand_lvl = random.choice(['Low', 'Medium', 'High'])
        
    with open(state_logs_out, "wb") as f:
        pickle.dump(state_logs, f)
    print(f"Saved State Space Logs to {state_logs_out}")
    print("Training Pipeline Complete!")

if __name__ == "__main__":
    train_agent()
