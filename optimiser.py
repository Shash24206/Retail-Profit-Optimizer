from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a proper secret key for session management
bcrypt = Bcrypt(app)

# Users with hashed passwords for secure login
users = {'hruthik': bcrypt.generate_password_hash('1234').decode('utf-8'),
         'sushanth': bcrypt.generate_password_hash('6789').decode('utf-8')}

# Data processing
file_path = r"supermarket_sales.csv"
data = pd.read_csv(file_path)

# Handling missing values
if data.isnull().sum().any():
    data.fillna(data.mean(), inplace=True)  # Simple mean imputation

# Feature Engineering: Calculating profit
data['Profit'] = (data['gross margin percentage'] / 100) * data['Unit price'] * data['Quantity']
features = ['Unit price', 'Quantity', 'gross margin percentage']
X = data[features]
y = data['Profit']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Cross-Validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"Cross-validated MSE: {cv_scores.mean()}")

# Model evaluation on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")

# Add predicted profit to dataset
data['Predicted Profit'] = model.predict(X_scaled)
recommended_products = data.sort_values(by='Predicted Profit', ascending=False)

# Helper Functions
def login_required(f):
    """Decorator to check if user is logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def recommend_within_budget_for_line(product_line, allocated_budget):
    """Recommend products within the allocated budget for a specific product line."""
    recommended_products_within_budget = []
    total_cost = 0
    filtered_data = recommended_products[recommended_products['Product line'] == product_line]

    for index, row in filtered_data.iterrows():
        product_total = row['Unit price'] * row['Quantity']
        if total_cost + product_total <= allocated_budget:
            recommended_products_within_budget.append(row)
            total_cost += product_total
        elif total_cost < allocated_budget:
            remaining_budget = allocated_budget - total_cost
            max_quantity_affordable = int(remaining_budget // row['Unit price'])
            if max_quantity_affordable > 0:
                row['Quantity'] = max_quantity_affordable
                recommended_products_within_budget.append(row)
                total_cost += max_quantity_affordable * row['Unit price']
        if total_cost >= allocated_budget:
            break
    return pd.DataFrame(recommended_products_within_budget)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) and bcrypt.check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password, please try again.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/')
@login_required
def index():
    """Main page."""
    return render_template('index.html', data=data)

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    """Recommendation page where user chooses budget and priority."""
    budget = float(request.form['budget'])
    priority_dict = {}

    product_lines = data['Product line'].unique()
    for line in product_lines:
        priority = float(request.form.get(f"priority_{line}", 0))
        priority_dict[line] = priority

    # Normalize priorities
    total_priority = sum(priority_dict.values())
    for line in priority_dict:
        priority_dict[line] = priority_dict[line] / total_priority

    recommended_products_within_budget_all_lines = []
    for line, priority in priority_dict.items():
        allocated_budget = budget * priority
        recommended_df_line = recommend_within_budget_for_line(line, allocated_budget)
        recommended_products_within_budget_all_lines.append(recommended_df_line)

    recommended_df_all_lines = pd.concat(recommended_products_within_budget_all_lines)

    # Calculate the cost percentage and profit percentage for recommendations
    total_cost_of_recommendations = recommended_df_all_lines['Quantity'] * recommended_df_all_lines['Unit price']
    recommended_df_all_lines['Percentage of Budget'] = (total_cost_of_recommendations / budget) * 100
    recommended_df_all_lines['Profit Percentage'] = (recommended_df_all_lines['Profit'] / total_cost_of_recommendations) * 100

    grouped_budget = recommended_df_all_lines.groupby('Product line').agg(
        total_quantity=('Quantity', 'sum'),
        total_cost=('Percentage of Budget', 'sum'),
        total_profit_percentage=('Profit Percentage', 'sum')
    ).reset_index()

    # Calculate total profit percentage for all recommendations
    total_profit = recommended_df_all_lines['Profit'].sum()
    total_recommendation_cost = total_cost_of_recommendations.sum()
    total_profit_percentage = (total_profit / total_recommendation_cost) * 100

    return render_template('recommendations.html', recommendations=grouped_budget, total_profit_percentage=total_profit_percentage)

if __name__ == '__main__':
    app.run(debug=True)
