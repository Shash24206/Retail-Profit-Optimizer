# Retail-Profit-Optimizer : Smart Product Recommendations and Budget Optimization

## **Features**

### User Authentication
- Secure login functionality for market owners using Flask sessions, ensuring sensitive data is protected.

### Dynamic Product Recommendations
- Predicts product profitability using a linear regression model based on features like unit price, quantity, and gross margin percentage.
- Recommends products within a specified budget, considering priorities for different product lines.

### Machine Learning Model
- Trains a linear regression model using historical sales data to predict profitability.
- Provides insights on which products will generate the most profit for retailers.

### Priority-Based Budget Allocation
- Allows users to assign priorities to product lines (e.g., groceries, electronics).
- Allocates the budget across product lines proportionally for the most optimal recommendations.

### Visualization & Insights
- Displays recommendations grouped by product line, showing cost and profit percentage for each.
- Offers insights into the cost-effectiveness of product recommendations within the allocated budget.

## **Tech Stack**

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn
- **Data Handling:** Pandas
- **Frontend:** HTML, CSS, JavaScript (Flask templates)
- **Data Source:** CSV file with historical sales data

## **Project Structure**

- **app.py:** Main Flask application file handling all routes and logic.
- **supermarket_sales.csv:** Dataset used to train the machine learning model for product recommendations.
- **templates/:** Folder containing HTML templates for the web application.
  - **login.html:** Login page for user authentication.
  - **index.html:** Dashboard for entering budget and priority settings.
  - **recommendations.html:** Page displaying product recommendations and insights.

## **Usage**

### 1. Log In
- Enter your credentials to access the dashboard.

### 2. Input Budget and Priorities
- Specify your total budget.
- Assign priorities to product lines such as groceries, electronics, etc.

### 3. View Recommendations
- Receive product recommendations based on your budget and priorities, including cost and profit percentage insights for each product line.

### 4. Download Results (Optional)
- Future functionality will include the option to export recommendations as a CSV file.

## **Future Enhancements**

- **Advanced Machine Learning Models:**
  - Implement more advanced models like Random Forest or XGBoost to improve prediction accuracy.

- **Real-Time Data:**
  - Integrate APIs to fetch live data on demand and stock levels for more accurate recommendations.

- **Improved User Management:**
  - Use a database for user authentication, improving security and scalability.

- **Enhanced Visualization:**
  - Add interactive graphs using tools like Plotly or Matplotlib for better data analysis and visualization.

- **CSV Export:**
  - Add functionality to download the recommendations and budget allocation data in CSV format for offline use.


