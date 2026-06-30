import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')


# Load data
print("Loading data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Get feature names
feature_names = X_train.columns.tolist()
n_features = len(feature_names)

print(f"\nTotal number of features: {n_features}")
print(f"Target: Select 5 best features using Genetic Algorithm\n")


# GA Parameters
N_FEATURES_TO_SELECT = 5
POPULATION_SIZE = 50
N_GENERATIONS = 30
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)


# Define fitness function
def evaluate_features(individual):
    """
    Evaluate a subset of features using RandomForest regression.
    Returns a tuple (r2_score, -mae) for multi-objective optimization.
    """
    # Get selected features (where individual has 1)
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    
    # Must have at least 1 feature and exactly N_FEATURES_TO_SELECT features
    if len(selected_features) != N_FEATURES_TO_SELECT:
        # Return very poor fitness if constraint not met
        return (-1000.0, -1000.0)
    
    # Select features from training and test data
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    # Train RandomForest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_selected)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Return r2 (maximize) and negative MAE (maximize = minimize MAE)
    return (r2, -mae)


# Create DEAP types
# We want to maximize both objectives (r2 and -mae)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize DEAP toolbox
toolbox = base.Toolbox()


# Custom initialization to ensure exactly N_FEATURES_TO_SELECT features are selected
def create_individual():
    """Create an individual with exactly N_FEATURES_TO_SELECT features selected."""
    individual = [0] * n_features
    selected_indices = random.sample(range(n_features), N_FEATURES_TO_SELECT)
    for idx in selected_indices:
        individual[idx] = 1
    return creator.Individual(individual)


# Register genetic operators
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_features)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


def fix_individual(individual):
    """Ensure individual has exactly N_FEATURES_TO_SELECT features selected."""
    selected_count = sum(individual)
    
    if selected_count < N_FEATURES_TO_SELECT:
        # Add random features
        available = [i for i, bit in enumerate(individual) if bit == 0]
        to_add = random.sample(available, N_FEATURES_TO_SELECT - selected_count)
        for idx in to_add:
            individual[idx] = 1
    elif selected_count > N_FEATURES_TO_SELECT:
        # Remove random features
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        to_remove = random.sample(selected, selected_count - N_FEATURES_TO_SELECT)
        for idx in to_remove:
            individual[idx] = 0
    
    return individual


def custom_mate(ind1, ind2):
    """Custom crossover that maintains feature count constraint."""
    tools.cxTwoPoint(ind1, ind2)
    fix_individual(ind1)
    fix_individual(ind2)
    return ind1, ind2


def custom_mutate(individual):
    """Custom mutation that maintains feature count constraint."""
    # Swap a selected feature with an unselected one
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    unselected = [i for i, bit in enumerate(individual) if bit == 0]
    
    if len(selected) > 0 and len(unselected) > 0 and random.random() < MUTATION_PROB:
        swap_out = random.choice(selected)
        swap_in = random.choice(unselected)
        individual[swap_out] = 0
        individual[swap_in] = 1
    
    return individual,


# Update toolbox with custom operators
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)


# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)


# Run Genetic Algorithm
print("="*60)
print("Starting Genetic Algorithm for Feature Selection")
print("="*60)
print(f"Population Size: {POPULATION_SIZE}")
print(f"Generations: {N_GENERATIONS}")
print(f"Crossover Probability: {CROSSOVER_PROB}")
print(f"Mutation Probability: {MUTATION_PROB}")
print(f"Tournament Size: {TOURNAMENT_SIZE}")
print("="*60)
print()

# Create initial population
population = toolbox.population(n=POPULATION_SIZE)

# Evaluate initial population
print("Evaluating initial population...")
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

print(f"Initial population evaluated.\n")

# Evolution loop
for gen in range(N_GENERATIONS):
    print(f"Generation {gen + 1}/{N_GENERATIONS}")
    
    # Select offspring
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # Apply mutation
    for mutant in offspring:
        if random.random() < MUTATION_PROB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate individuals with invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Replace population
    population[:] = offspring
    
    # Gather statistics
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_r2, best_neg_mae = best_ind.fitness.values
    
    print(f"  Best R2: {best_r2:.4f}, Best MAE: {-best_neg_mae:.4f}")
    print(f"  Avg R2: {record['avg'][0]:.4f}, Avg MAE: {-record['avg'][1]:.4f}")
    print()

print("="*60)
print("Genetic Algorithm Complete!")
print("="*60)
print()

# Get the best individual
best_individual = tools.selBest(population, 1)[0]
selected_feature_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
selected_feature_names = [feature_names[i] for i in selected_feature_indices]

print("Best Feature Subset Found:")
print("-" * 60)
for i, (idx, name) in enumerate(zip(selected_feature_indices, selected_feature_names)):
    print(f"  {i+1}. Feature {idx}: {name}")
print()

# Train final model with selected features
print("Training final RandomForest model with selected features...")
X_train_best = X_train.iloc[:, selected_feature_indices]
X_test_best = X_test.iloc[:, selected_feature_indices]

final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_best, y_train)

# Make predictions
y_train_pred = final_model.predict(X_train_best)
y_test_pred = final_model.predict(X_test_best)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print()
print("="*60)
print("Final Model Performance")
print("="*60)
print(f"Training Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE: {train_mae:.4f}")
print()
print(f"Test Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE: {test_mae:.4f}")
print("="*60)

# Feature importance from the final model
feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print()
print("Feature Importance (from RandomForest):")
print("-" * 60)
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
print("="*60)

print("\nFeature selection complete!")

