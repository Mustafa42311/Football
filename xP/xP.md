## Feature Engineering
### Pass Length (Euclidean Distance)

```python
events['pass_distance'] = np.sqrt((events['end_x'] - events['x'])**2 + (events['end_y'] - events['y'])**2)
```

This applies the **Pythagorean Theorem** to calculate the **Euclidean Distance** between two points on a 2D plane.

$$
d=(x_2−x_1)^2+(y_2−y_1)^2
$$

### Pass Angle (Directionality)

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.28em" viewBox="0 0 400000 1296" preserveAspectRatio="xMinYMin slice"><path d="M263,681c0.7,0,18,39.7,52,119
c34,79.3,68.167,158.7,102.5,238c34.3,79.3,51.8,119.3,52.5,120
c340,-704.7,510.7,-1060.3,512,-1067
l0 -0
c4.7,-7.3,11,-11,19,-11
H40000v40H1012.3
s-271.3,567,-271.3,567c-38.7,80.7,-84,175,-136,283c-52,108,-89.167,185.3,-111.5,232
c-22.3,46.7,-33.8,70.3,-34.5,71c-4.7,4.7,-12.3,7,-23,7s-12,-1,-12,-1
s-109,-253,-109,-253c-72.7,-168,-109.3,-252,-110,-252c-10.7,8,-22,16.7,-34,26
c-22,17.3,-33.3,26,-34,26s-26,-26,-26,-26s76,-59,76,-59s76,-60,76,-60z
M1001 80h400000v40h-400000z"></path></svg>)

```python
events['pass_angle'] = np.arctan2(events['end_y'] - events['y'], events['end_x'] - events['x'])
```

This uses the **2-argument arctangent** function (`arctan2`). Unlike the standard `arctan`, which only works for two quadrants (−2π to 2π), `arctan2(dy, dx)` handles all four quadrants correctly, returning an angle in the range (−π,π].

$$
θ=arctan(\frac{y_{end}−y_{start}}{x_{end}−x_{start}})
$$

It tells the model the **direction of play**.

A backward pass (angle ≈π or 180∘) is usually safe and easy. A forward vertical pass (angle ≈0) is often blocked by opposition lines. Without this, the model wouldn't know if a pass was progressive or defensive.

### Pitch Symmetry (Mirrored Y)

```python
events['mirrored_y'] = abs(events['y'] - 40)
```

 The data uses a coordinate system where the pitch width is 80 (e.g., StatsBomb), making y=40 the center.

$$
∣y_{coordinate}−y_{center}∣
$$

**Normalization:** In football, the difficulty of a pass from the left wing is statistically identical to a pass from the right wing.

 By converting the y coordinate (0 to 80) into a "distance from center" (0 to 40), we treat the left and right flanks as the same spatial feature. This reduces noise and helps the model generalize better with fewer data points (it doesn't have to "re-learn" that the sidelines are dangerous on both sides).

### Sequence Depth (Pass Count in Possession)

```python
events['pass_count_in_possession'] = events.groupby(['match_id', 'possession'])['is_pass'].cumsum() - 1
```

**`groupby(['match_id', 'possession'])`**: This isolates every unique "chain" of possession. We don't want a pass from the 5th minute affecting the count of a possession in the 80th minute, so we treat each possession as its own mini-dataset.

**`.cumsum()` (Cumulative Sum)**: This iterates through the events in chronological order, adding `1` every time it encounters a pass.

- **`1` (The Offset)**: Since `cumsum` includes the *current* row, the first pass would be labeled "1". By subtracting 1, we convert this into **"Preceding Passes."**
    - *First Pass:* 1−1=0 (0 passes came before this).
    - *Second Pass:* 2−1=1 (1 pass came before this)
- **Defensive Disorganization:** A pass made after 20 consecutive passes (sustained pressure) faces a different defensive structure than the 1st pass of a counter-attack. Long passing chains often drag defenders out of position.
- **Fatigue & Focus:** As the pass count rises, the probability of a defensive error typically increases, potentially impacting the xP (Expected Pass) value.

### Pre-Pass Movement (Carry Distance)

```python
events['carry_length'] = np.sqrt(
    (events['carry_end_location'].apply(lambda x: x[0]...) - events['location'].apply(lambda x: x[0]...))**2 + 
    (events['carry_end_location'].apply(lambda x: x[1]...) - events['location'].apply(lambda x: x[1]...))**2
)
```

- **`.apply(lambda x: x[0] ...)`**: The raw data  stores locations as lists `[x, y]` inside a single column. You cannot do math directly on a list. This lambda function "unpacks" the list to access the `x` (index 0) and `y` (index 1) coordinates individually.
- **`np.nan`**: Handles cases where there is no carry data (e.g., a first-touch pass), preventing the code from crashing
- Once extracted, it applies the standard **Euclidean Distance** formula again.

 A long carry moves the defensive block. If a player carries the ball 20 meters, defenders have to leave their zones to engage, often opening up passing lanes that didn't exist before.

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.28em" viewBox="0 0 400000 1296" preserveAspectRatio="xMinYMin slice"><path d="M263,681c0.7,0,18,39.7,52,119
c34,79.3,68.167,158.7,102.5,238c34.3,79.3,51.8,119.3,52.5,120
c340,-704.7,510.7,-1060.3,512,-1067
l0 -0
c4.7,-7.3,11,-11,19,-11
H40000v40H1012.3
s-271.3,567,-271.3,567c-38.7,80.7,-84,175,-136,283c-52,108,-89.167,185.3,-111.5,232
c-22.3,46.7,-33.8,70.3,-34.5,71c-4.7,4.7,-12.3,7,-23,7s-12,-1,-12,-1
s-109,-253,-109,-253c-72.7,-168,-109.3,-252,-110,-252c-10.7,8,-22,16.7,-34,26
c-22,17.3,-33.3,26,-34,26s-26,-26,-26,-26s76,-59,76,-59s76,-60,76,-60z
M1001 80h400000v40h-400000z"></path></svg>)

- **Static vs. Dynamic:** A pass made from a standing position (0 carry) is harder to execute than one made after a carry, as the latter allows the passer to generate more momentum.

### Time Normalization

```python
events['time_seconds'] = events['minute']*60 + events['second']
```

A simple unit conversion to linear time.

$$
t_{total}=(t_{min}×60)+t_{sec}
$$

**Continuous Variable:** Machine learning models struggle with "Clock Time" (e.g., "45:30") because it is not a continuous scalar. Converting to total seconds allows us to calculate time deltas (Δt) effortlessly.

### Speed of Play (Vertical Velocity)

```python
events['vertical_velocity'] = (events['x'] - events['x'].shift(3)) / (events['time_seconds'] - events['time_seconds'].shift(3))
```

This approximates the velocity of the ball's movement along the x-axis (down the field) using a **Finite Difference** method over a window of 3 events.

$$
v_x≈\frac{Δx}{Δt}=\frac{x_i−x_{i-3}}{t_i−t_{i−3}}
$$

- **Why `.shift(3)`? (Smoothing):**
    - Using `.shift(1)` (the immediate previous event) can be noisy. A player might make a tiny touch to set themselves up, which would result in near-zero velocity.
    - By looking back **3 events**, we capture the **average trend** of the play. It answers: "Over the last few actions, how fast are we approaching the opponent's goal?"
- **Counter-Attacks vs. Possession:** High positive vertical velocity indicates a rapid counter-attack (high xP, but high execution difficulty). Low or negative velocity indicates slow build-up or recycling possession.
- **Defensive State:** Defenses are more vulnerable when the ball is moving quickly towards them (high velocity) compared to when the attack is static.

## **XGBClassifier**.

To understand XGBoost (Extreme Gradient Boosting), you have to understand the three concepts that build it, from simple to complex.

### The Building Block: The Decision Tree

At its heart, XGBoost is just a bunch of "Decision Trees." A decision tree is like a game of **20 Questions**.

- *Is the animal big?* (Yes/No)
- *Does it have a trunk?* (Yes/No)
- *Prediction:* It's an Elephant.

A single tree is easy to understand, but it's usually not very smart on its own. It makes mistakes easily.

### The Strategy: "Boosting"

This is the most important part. In traditional machine learning, you might train 100 trees separately and average their answers. That is called "Bagging" (like a Democracy).

**Boosting** is different. It is a **Sequential** process (like a relay race).

1. **Tree 1** makes a prediction. It gets some right, some wrong.
2. **Tree 2** is created, but it is explicitly told: *"Don't worry about the easy ones Tree 1 got right. Focus ONLY on the ones Tree 1 got wrong."*
3. **Tree 3** focuses on the ones Tree 2 still got wrong.

This cycle repeats. Each new model corrects the errors of the previous one.

### The "Extreme" Part (XGBoost)

"Gradient Boosting" is the math behind step #2. **XGBoost** stands for **"Extreme Gradient Boosting."**

The "Extreme" doesn't mean it's radical; it means it is **computationally optimized** for speed and performance.

- **Speed:** It can build these trees in parallel (using all your computer's cores at once), making it much faster than older boosting methods.
- **Handling Missing Data:** If your data has holes in it (NaNs), XGBoost automatically learns the best way to handle them. You don't always have to fill them in manually.
- **Regularization:** This is a fancy term for "preventing the model from memorizing the data." XGBoost has built-in penalties to keep the model simple and flexible, so it works well on new data it hasn't seen before.
