import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# Set up terminal printing options for cleaner output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
np.set_printoptions(precision=4)



# Define the Google Sheet URL
gsheet_url = "https://docs.google.com/spreadsheets/d/1VK1jbJcaaCL9vFHKGsBZduaG5-yjjUq1/edit?usp=sharing&ouid=105139407976399453589&rtpof=true&sd=true"

# Convert the Google Sheet URL to the direct CSV export link
csv_url = gsheet_url.replace("/edit?usp=sharing", "/export?format=csv")

print("--- Step 1: Loading Data from Google Sheet ---")

    # Read the data directly as CSV
penalty_kick_df = pd.read_csv(csv_url, header=0)
    
 
# Clean up and select relevant columns (Outcome must be 1 for Goal, 0 for Miss/Saved)
columns_to_keep = ['Kicker_Side', 'Goalie_Side', 'Outcome']
penalty_kick_df = penalty_kick_df[columns_to_keep]


# --- 2. BUILD THE PAYOFF MATRIX (KICKER'S EXPECTED VALUE) ---

# Calculate the mean 'Outcome' (Scoring Probability) for each 3x3 combination
payoff_matrix = penalty_kick_df.groupby(['Kicker_Side', 'Goalie_Side'])['Outcome'].mean()
print("\n--- Step 2: Payoff Matrix (Kicker's Scoring Probability) ---")
print(payoff_matrix.to_string()) # Using to_string() for clean terminal output

print("\nSample Size (Count) for each of the 9 buckets:")
bucket_counts = penalty_kick_df.groupby(['Kicker_Side', 'Goalie_Side']).size()
print(bucket_counts.to_string())

# Reshape the Series into a 3x3 grid (DataFrame)
payoff_grid = payoff_matrix.unstack()
# Convert the DataFrame grid to a NumPy array for the solver
P_matrix = payoff_grid.values


# --- 3. SOLVE FOR NASH EQUILIBRIUM (KICKER'S OPTIMAL MIXED STRATEGY) ---

# The Nash Equilibrium solves for the kicker's probabilities (k_C, k_L, k_R)
# that make the Goalie indifferent to diving C, L, or R.


print("\n--- Step 3: Solving Nash Equilibrium (Ax = b) ---")
    
    # Kicker order in P_matrix rows is: C, L, R
    # Goalie order in P_matrix columns is: C, L, R
    
    # Setup for solving Ax=b where x = [k_C, k_L, k_R]
A_kicker = np.array([
        [P_matrix[0,0]-P_matrix[0,1], P_matrix[1,0]-P_matrix[1,1], P_matrix[2,0]-P_matrix[2,1]], # Eq 1: E(Goalie C) = E(Goalie L)
        [P_matrix[0,1]-P_matrix[0,2], P_matrix[1,1]-P_matrix[1,2], P_matrix[2,1]-P_matrix[2,2]], # Eq 2: E(Goalie L) = E(Goalie R)
        [1, 1, 1]                                                                                # Eq 3: Probs must sum to 1
    ])
    
    # 'b' is the answer vector [0, 0, 1]
b_kicker = np.array([0, 0, 1])
    
    # Solve the system
kicker_strategy = np.linalg.solve(A_kicker, b_kicker)
    
print("\n--- Kicker's Optimal Strategy (Nash Equilibrium) ---")
print(f"Shoot Centre (C): {kicker_strategy[0]:.2%}")
print(f"Shoot Left (L):   {kicker_strategy[1]:.2%}")
print(f"Shoot Right (R):  {kicker_strategy[2]:.2%}")


# --- 4. CALCULATE ACTUAL KICKER STRATEGY (REALITY CHECK) ---

print("\n--- Actual Kicker Strategy (From Real Data) ---")
actual_kicker_strategy = penalty_kick_df['Kicker_Side'].value_counts(normalize=True)
print(actual_kicker_strategy.to_string(float_format='%.4f'))


# --- 5. VISUALIZATION (COMPARISON CHART) ---

categories = ['Left', 'Right', 'Centre']
bar_width = 0.35

# 5a. Prepare Model Strategy (kicker_strategy is currently [C, L, R])
# We need to reorder to match categories: [L, R, C]
model_probs_LRC = [kicker_strategy[1], kicker_strategy[2], kicker_strategy[0]]

# 5b. Prepare Actual Strategy (reindex to ensure L, R, C order)
actual_probs_LRC = actual_kicker_strategy.reindex(['L', 'R', 'C']).fillna(0).tolist()


fig, ax = plt.subplots(figsize=(10, 6))

# Plot Model Strategy
rects1 = ax.bar(np.arange(len(categories)) - bar_width/2,
                model_probs_LRC,
                bar_width,
                label='Model (Nash Equilibrium)',
                color='#1f77b4') # Blue

# Plot Actual Strategy
rects2 = ax.bar(np.arange(len(categories)) + bar_width/2,
                actual_probs_LRC,
                bar_width,
                label='Actual (Observed Data)',
                color='#ff7f0e') # Orange

# Add labels and titles
ax.set_ylabel('Probability (%)', fontsize=12)
ax.set_title('Optimal (Model) vs. Actual Kicker Strategy', fontsize=16, pad=20)
ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(categories, fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels on top of the bars
def autolabel(rects, label_format):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(label_format % (height * 100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

autolabel(rects1, '%.1f%%')
autolabel(rects2, '%.1f%%')

# Set y-axis limit slightly above 1.0 for better visibility of labels
ax.set_ylim(0, max(max(model_probs_LRC), max(actual_probs_LRC)) * 1.1)

# Display the plot
print("\n--- Step 4: Displaying Comparison Plot ---")
plt.show()

# --- 6. OPTIONAL: HEATMAP OF PAYOFF MATRIX ---
plt.figure(figsize=(8, 7))
sns.heatmap(payoff_grid, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5, linecolor='black',
            cbar_kws={'label': 'Kicker Scoring Probability'})
plt.title('Penalty Kick Payoff Matrix Heatmap', fontsize=16)
plt.xlabel('Goalie Side', fontsize=12)
plt.ylabel('Kicker Side', fontsize=12)
plt.show()
