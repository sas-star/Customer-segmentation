from flask import Flask, send_from_directory, jsonify
import os
import pickle
import numpy as np
import pandas as pd
import random

app = Flask(__name__, static_folder="../frontend", static_url_path="")

MODELS_DIR = "/Users/croma/Desktop/unsupervised_customer_segmentation/models"

def load_pickle(filename):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/stats")
def get_stats():
    stats = load_pickle("episode_stats.pkl")
    if stats is None:
        return jsonify({"error": "Stats model not found"}), 404
    return jsonify({
        "rewards": stats["rewards"],
        "steps": stats["steps"],
        "waste_reduction": stats["waste_reduction"],
        "profits": stats["profits"]
    })

@app.route("/api/q_table")
def get_q_table():
    q_table = load_pickle("q_table.pkl")
    if q_table is None:
        return jsonify({"error": "Q-table model not found"}), 404
        
    formatted_q = {}
    for key, val in q_table.items():
        str_key = f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"
        formatted_q[str_key] = val
        
    return jsonify(formatted_q)

@app.route("/api/predict")
def get_predict():
    # Load dataset & policy
    df = load_pickle("cleaned_food_data.pkl")
    policy = load_pickle("policy.pkl")
    
    if df is None:
        return jsonify({"error": "Cleaned dataset pickle not found"}), 404
    if policy is None:
        return jsonify({"error": "Policy pickle not found"}), 404
        
    # Get base prices for categories
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
    
    # Category list and their respective median limits to discretize
    categories = df['category'].unique()
    medians = {}
    for cat in categories:
        median_val = df[df['category'] == cat]['value'].median()
        if pd.isna(median_val) or median_val == 0:
            median_val = 100000.0
        medians[cat] = median_val
        
    def discretize_stock(val, median_val):
        if val < 0.5 * median_val:
            return 'Low'
        elif val < 1.5 * median_val:
            return 'Medium'
        else:
            return 'High'
            
    # Sample 800 rows from the dataset to make a realistic batch evaluation
    sample_df = df.sample(min(800, len(df)), random_state=42).copy()
    
    # Track statistics
    q_profits = []
    base_profits = []
    actions_count = {0: 0, 1: 0, 2: 0, 3: 0} # Maintain, Discount, Refrigerate, Repurpose
    
    # Track supply, demand, sold details for first 50 steps
    supply_levels = []
    demand_levels = []
    sold_levels = []
    
    # Track waste per category
    cat_q_waste = {cat: 0.0 for cat in categories}
    cat_base_waste = {cat: 0.0 for cat in categories}
    
    q_cum_profit = 0.0
    base_cum_profit = 0.0
    
    q_total_waste = 0.0
    base_total_waste = 0.0
    
    for idx, row in sample_df.iterrows():
        country = row['country_or_area']
        cat = row['category']
        val = row['value']
        
        # Discretize stock level
        median_val = medians.get(cat, 100000.0)
        stock_lvl = discretize_stock(val, median_val)
        stock_val = 1 if stock_lvl == 'Low' else (2 if stock_lvl == 'Medium' else 3)
        
        # Simulate a demand state
        demand_lvl = random.choice(['Low', 'Medium', 'High'])
        demand_val = 1 if demand_lvl == 'Low' else (2 if demand_lvl == 'Medium' else 3)
        
        # Look up policy
        state_key = (country, cat, stock_lvl, demand_lvl)
        action = 0
        if state_key in policy:
            action = int(np.argmax(policy[state_key]))
        else:
            # Intuitive fallback logic
            if stock_lvl == 'High' and demand_lvl == 'Low':
                action = 1 # Discount
            elif stock_lvl == 'High' and demand_lvl == 'Medium':
                action = 3 # Repurpose
            elif stock_lvl == 'Medium' and demand_lvl == 'Low':
                action = 2 # Refrigerate
            else:
                action = 0 # Maintain
                
        actions_count[action] += 1
        base_p = base_prices.get(cat, 10.0)
        
        # --- Q-LEARNING EVALUATION ---
        if action == 1:
            adj_demand = min(3, demand_val + 1)
        elif action == 3:
            adj_demand = min(3.0, demand_val + 0.5)
        else:
            adj_demand = demand_val
            
        q_sold = min(float(stock_val), float(adj_demand))
        q_rem = max(0.0, float(stock_val) - q_sold)
        
        q_spoil_rate = 0.1 if action == 2 else (0.15 if action == 3 else 0.4)
        q_waste = q_rem * q_spoil_rate
        q_total_waste += q_waste
        cat_q_waste[cat] += q_waste
        
        q_price = base_p * 0.85 if action == 1 else (base_p * 0.90 if action == 3 else base_p)
        q_revenue = q_sold * q_price
        
        q_spoil_cost = q_waste * (base_p * 0.6)
        q_hold_cost = base_p * 0.15 if action == 2 else 0.0
        q_repurpose_cost = base_p * 0.08 if action == 3 else 0.0
        q_profit = q_revenue - (q_spoil_cost + q_hold_cost + q_repurpose_cost)
        q_cum_profit += q_profit
        q_profits.append(q_cum_profit)
        
        # --- BASELINE EVALUATION (Action 0 - Standard) ---
        base_sold = min(float(stock_val), float(demand_val))
        base_rem = max(0.0, float(stock_val) - base_sold)
        base_waste = base_rem * 0.4
        base_total_waste += base_waste
        cat_base_waste[cat] += base_waste
        
        base_revenue = base_sold * base_p
        base_spoil_cost = base_waste * (base_p * 0.6)
        base_profit = base_revenue - base_spoil_cost
        base_cum_profit += base_profit
        base_profits.append(base_cum_profit)
        
        # Record first 50 entries for supply/demand chart
        if len(supply_levels) < 50:
            supply_levels.append(stock_val)
            demand_levels.append(demand_val)
            sold_levels.append(q_sold)
            
    # Format action counts
    action_labels = ["Maintain Standard", "Discount Price (-15%)", "Refrigerate Surplus", "Repurpose Surplus"]
    action_dist = {action_labels[k]: v for k, v in actions_count.items()}
    
    # Filter waste per category for top 5 categories
    top_categories = sorted(cat_base_waste.keys(), key=lambda x: cat_base_waste[x], reverse=True)[:5]
    waste_by_cat = {
        "categories": top_categories,
        "q_learning": [round(cat_q_waste[c], 2) for c in top_categories],
        "baseline": [round(cat_base_waste[c], 2) for c in top_categories]
    }
    
    # Reduction percentage
    waste_reduction_pct = 0.0
    if base_total_waste > 0:
        waste_reduction_pct = round(((base_total_waste - q_total_waste) / base_total_waste) * 100, 1)
        
    return jsonify({
        "profits_comparison": {
            "index": list(range(1, len(q_profits) + 1)),
            "q_learning": [round(p, 2) for p in q_profits],
            "baseline": [round(p, 2) for p in base_profits]
        },
        "action_distribution": action_dist,
        "supply_demand_match": {
            "index": list(range(1, len(supply_levels) + 1)),
            "supply": supply_levels,
            "demand": demand_levels,
            "sold": sold_levels
        },
        "waste_by_category": waste_by_cat,
        "summary": {
            "q_learning_profit": round(q_cum_profit, 2),
            "baseline_profit": round(base_cum_profit, 2),
            "q_learning_waste": round(q_total_waste, 2),
            "baseline_waste": round(base_total_waste, 2),
            "waste_reduction_pct": waste_reduction_pct
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5002)
