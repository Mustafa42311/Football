# xG Model Mathematics
I will be explaing here what is Logistics Regression and how does it work and the Euclidean distance (how I calculated the distance between the shot and the goal) and the metrics I got on my model and what do they actually mean.

## Logistic Regression
Logistic Regression is a binary model it will predicts a continuous number, it is used to predict the probabilty of an event happening.
Logistic regression takes a linear equation (y=mx+b) and wraps it inside the Sigmoid Function:

$$\text{P}_{\text{Goal}} = \frac{1}{1+e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots)}}$$

where $$\beta_0$$, $$\beta_1$$ and etc. are diffrent inputs in the model (Distance, Angle, Header, etc.) and assigns them "weights" (importance).

Example: Distance gets a negative weight (further away = less chance).

That is ho xG is calculated using Logistic Regression.


## Euclidean Distance
Euclidean Distance is a simple Pythagorean Theorem $$(a^2 + b^2=c^2)$$ applied to a coordinate system.

![WhatsApp Image 2025-12-10 at 6 17 39 PM](https://github.com/user-attachments/assets/fddd3d22-e7fe-49b9-b604-9335748563a6)

- Point A (The Shot): Your player is at coordinates (x,y).

- Point B (The Goal): The center of the goal is fixed at (100,50) (in Opta coordinates).

- Side a (Length difference): (df['x'] - 100). This measures how far down the field the player is from the goal line.

- Side b (Width difference): (df['y'] - 50). This measures how far to the left or right the player is from the center of the pitch.

- Hypotenuse c (The Distance): To find the length of the direct path to goal, we square both sides, add them, and take the square root.

## Model Metrics

### Log Loss ( Confidence Score )

If a shot is a Goal (1) and you predict 0.95, the loss is tiny (Good job!), but if a shot is a Goal (1) and you predict 0.01, the loss is huge.
The lower the score the better, 0 is perfect.

### ROC AUC ( Receiver Operating Characteristic - Area Under Curve )

Grabs a random goal and a random miss, if the xG of the goal was higher than the miss, the model takes points. The higher the better.

### Brier Score ( Precision )

Measures the mean squared difference between your prediction and the actual outcome.

$$(\text{Result} - \text{Prediction})^2$$

The lower the better.
