# AI-Based Traffic Congestion Management System

An end-to-end machine-learning prototype for adaptive traffic signal control. The project generates time-based traffic observations, engineers traffic-flow features, trains three complementary ML models, and saves the trained components for use in a traffic-light simulation.

The main objective is to move beyond fixed signal timings and explore how traffic density, direction-wise demand, and recurring traffic patterns can be used to recommend dynamic green-light durations.

## Project Overview

The system combines three machine-learning tasks:

- **Classification:** A Random Forest classifies traffic density as Low, Medium, or High.
- **Clustering:** K-Means discovers recurring traffic patterns without predefined labels.
- **Regression:** A Gradient Boosting Regressor predicts the North-South green-light duration.

The current repository contains the reproducible ML pipeline and trained model files. The traffic simulator is referenced as a separate Git submodule and must be restored before it can be run from this repository.

## Pipeline

```text
Synthetic Traffic Generation
           |
           v
Data Cleaning and Feature Engineering
           |
           +-------------------+
           |                   |
           v                   v
 Random Forest             K-Means
 Density Classifier     Pattern Clustering
           |                   |
           +---------+---------+
                     |
                     v
       Gradient Boosting Regressor
          Green-Time Prediction
                     |
                     v
          Saved Models for Simulation
```

## Current Dataset

`generate_data.py` creates synthetic traffic observations for 90 days at two-minute intervals:

```text
90 days x 24 hours x 30 observations per hour = 64,800 records
```

Each observation includes:

- Timestamp, hour, and minute
- Day of the week and weekend flag
- North, South, East, and West vehicle counts
- North-South, East-West, and total vehicle counts

The generator represents common traffic behavior such as weekday morning rush, lunch traffic, evening rush, quieter night traffic, and different weekend patterns.

> **Scope note:** The accompanying academic report describes a larger experiment combining synthetic data with the UCI Metro Interstate Traffic Volume dataset. The current GitHub scripts do not contain the real-data merge logic, so this README documents the implementation that is directly reproducible from the repository.

## Feature Engineering

The preprocessing stage creates the following features:

| Feature | Description |
|---|---|
| `ns_total` | Combined North and South traffic |
| `ew_total` | Combined East and West traffic |
| `total_vehicles` | Total vehicles across all four directions |
| `ns_ratio` | North-South share of total traffic |
| `ew_ratio` | East-West share of total traffic |
| `hour_sin` | Sine encoding of the hour |
| `hour_cos` | Cosine encoding of the hour |
| `total_rolling_mean` | Rolling traffic mean over six observations |
| `total_rolling_std` | Rolling traffic standard deviation over six observations |
| `density_label` | Low, Medium, or High traffic category |
| `optimal_ns_green` | Prototype North-South green-time target |
| `optimal_ew_green` | Prototype East-West green-time target |

Cyclical hour encoding preserves the relationship between nearby times such as 23:00 and 00:00.

## Machine-Learning Models

### Random Forest Classifier

The classifier predicts traffic density using:

- Hour and day of week
- Weekend flag
- North-South and East-West totals
- Directional traffic ratios
- Cyclical hour features

The model uses 100 decision trees. It is evaluated using a classification report containing precision, recall, F1 score, support, and overall accuracy.

### K-Means Clustering

K-Means groups observations into four traffic patterns using:

- Hour
- Day of week
- Total vehicles
- North-South ratio
- East-West ratio

The features are standardized before clustering because K-Means uses distance. The fitted `StandardScaler` is saved with the clustering model so the same transformation can be applied during inference.

### Gradient Boosting Regressor

The regressor predicts the North-South green-light duration using traffic, time, and cluster features. It is evaluated with:

- **RMSE:** Prediction error expressed in seconds, with greater weight on large errors
- **R-squared:** Proportion of target variation explained by the model

## Project Structure

```text
traffic_congestion_management/
|
|-- generate_data.py          # Generates synthetic traffic data
|-- preprocess_data.py        # Cleans data and engineers features
|-- train_models.py           # Trains and evaluates all three models
|-- run_project.py            # Runs the pipeline and launches the simulator
|
|-- data/
|   |-- raw/
|   |   `-- traffic_data.csv
|   |-- traffic_data.csv
|   |-- processed_data.csv
|   `-- final_data.csv
|
|-- models/
|   |-- classifier.pkl
|   |-- clustering.pkl
|   |-- scaler.pkl
|   `-- regressor.pkl
|
`-- TrafficLightSimulation    # Simulator submodule reference
```

## Technologies and Libraries

### Verified in the current ML code

- Python
- Pandas
- NumPy
- scikit-learn
- joblib
- `datetime`, `random`, `os`, `sys`, and `subprocess`

### Described in the project report or simulator design

- Pygame for traffic visualization
- Matplotlib and Seaborn for charts
- OpenPyXL for Excel output

TensorFlow is not required by the models currently implemented in the GitHub training pipeline.

## Installation

Python 3.8 or later is recommended.

```bash
git clone https://github.com/Ravneet2653/traffic_congestion_management.git
cd traffic_congestion_management

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate
```

Install the ML dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

Install Pygame only after restoring the simulator:

```bash
pip install pygame
```

## Running the ML Pipeline

### 1. Generate traffic data

```bash
python generate_data.py
```

This creates `data/traffic_data.csv`.

### 2. Prepare the preprocessing input

The current preprocessor expects `data/raw/traffic_data.csv`. Until the scripts are updated to share one path, copy the generated file into the raw-data directory:

```bash
# macOS or Linux
mkdir -p data/raw
cp data/traffic_data.csv data/raw/traffic_data.csv
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force data/raw
Copy-Item data/traffic_data.csv data/raw/traffic_data.csv
```

### 3. Preprocess the data

```bash
python preprocess_data.py
```

This creates `data/processed_data.csv`.

### 4. Train the models

```bash
python train_models.py
```

This creates the final dataset and saves all model files inside `models/`.

## Saved Artifacts

| File | Purpose |
|---|---|
| `classifier.pkl` | Predicts Low, Medium, or High traffic density |
| `clustering.pkl` | Assigns an observation to a traffic-pattern cluster |
| `scaler.pkl` | Applies the scaling required by K-Means |
| `regressor.pkl` | Predicts the North-South green-light duration |

Models should be loaded only from trusted sources because pickle-based files can execute code during deserialization.

## Evaluation

The current code uses an 80/20 train-test split with `random_state=42`.

Classification output:

- Precision
- Recall
- F1 score
- Support
- Accuracy

Regression output:

- Root Mean Squared Error
- R-squared score

For stronger time-dependent evaluation, a future version should use chronological holdout data or `TimeSeriesSplit` instead of randomly mixing observations from different dates.

## Current Limitations

- The dataset generator is synthetic and cannot fully represent accidents, weather, road work, festivals, or real driver behavior.
- Density labels are created directly from total traffic, while related totals are also model inputs.
- The green-time target is calculated from `ns_ratio`, which is also a regression input. This can produce an overly optimistic R-squared value.
- The generated-data output path and preprocessing input path are different.
- Rolling features are generated but are not included in the current model feature lists.
- The code uses a random split for time-dependent observations.
- The repository does not currently calculate a confusion matrix or silhouette score.
- The simulator is stored as an unresolved Git submodule reference.
- A committed virtual environment makes the repository unnecessarily large and should be removed from version control.

These limitations mean that the project should be treated as an ML learning prototype rather than a production traffic-control system.

## Recommended Improvements

- Integrate real camera, loop-sensor, or open traffic data
- Create targets from observed waiting time or simulation-based optimization
- Use chronological validation and time-aware cross-validation
- Add confusion-matrix, silhouette-score, MAE, and baseline evaluation
- One-hot encode cluster membership before regression
- Add hyperparameter tuning with a leakage-safe pipeline
- Restore and document the Pygame simulator
- Add deterministic random seeds and automated tests
- Add a `requirements.txt` file and `.gitignore`
- Remove the committed virtual environment
- Extend the system to coordinate multiple intersections
- Add hard safety rules and a fixed-timing fallback before any real deployment

## Real-World Deployment Considerations

A real traffic-control system would require more than an ML prediction. It should include:

- Reliable camera or sensor inputs
- Minimum green, amber, and all-red timing rules
- Pedestrian crossing requirements
- Emergency-vehicle priority logic
- Latency and system-health monitoring
- Human override and a safe fixed-timing fallback
- Continuous monitoring for data drift and model degradation

## Key Learning

This project demonstrates how classification, clustering, and regression can be combined in one machine-learning pipeline. It also highlights an important ML lesson: strong scores are meaningful only when the target definition, validation strategy, and data source accurately represent the real problem.

## Contributors

- Ravneet Kaur
- Krishaay Sharma

## Academic Context

Machine Learning Project, Computer Science and Engineering Department, Thapar Institute of Engineering and Technology, Patiala.
