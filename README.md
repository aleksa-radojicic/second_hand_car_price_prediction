# Second-Hand Car Prices Prediction using Machine Learning Algorithms
This project was developed for my Bachelor's thesis at the [Faculty of Organizational Sciences](https://en.fon.bg.ac.rs/), Information Systems and Technologies. The goal was to predict prices of second-hand cars listed on the Serbian platform [*PolovniAutomobili*](https://www.polovniautomobili.com) using machine learning algorithms in Python.

## Project Overview
The objective of this project is to accurately predict prices of second-hand cars in euros using various ML models. The dataset consisted of 23,759 cars for training and 5,940 vehicles for testing. Coefficient of determination [R2] was the main metric used for models evaluation. Hyperparameters for Random Forest and Gradient Boosting Model were optimized using cross-validation. Furthermore, optimized GBMs were used to predict 90% prediction intervals.

GBM model performed best, achieving R2 of 0.94027 and MAE (mean absolute error) of 1230.770€ on test data. The prediction intervals contained the true values only 43.38% of the time.
The 0.50 quantile GBM model identified the production year as the most important attribute, followed by engine power, kilometerage, model name and others.

## Data Acquisition
Vehicles were scraped from the website [*PolovniAutomobili*](https://www.polovniautomobili.rs) using BeautifulSoup and Tor. SQLAlchemy and MySQL were used to store the data, resulting in a total of 30,788 rows and 50 columns.

## Technologies Used

- **Programming Language**: Python 3.11.8;
- **Database**: MySQL, SQLAlchemy;
- **Web Scraping**: BeautifulSoup, [Tor](https://www.torproject.org/), [Stem](https://stem.torproject.org/), Selenium, [tbselenium](https://pypi.org/project/tbselenium/);
- **Data Analysis**: Pandas, NumPy;
- **Visualization**: Matplotlib, Seaborn;
- **Machine Learning**: scikit-learn;
- **Configuration**: PyYAML, [Hydra](https://hydra.cc).


## Data Preprocessing
The Initial Cleaner component processed each column and removed outliers. The resulting data was passed then into a Preprocessor Pipeline, which prepared train and test set separately for modelling. Price (output column) was log-transformed to further improve model performance.

## Results
Below are the tables of performance metrics for the optimized models on both the training and test sets.

### Train Set Metrics

| Model | RMSE      | MAE       | R2       |
|-------|-----------|-----------|----------|
| RF    | **1191.538**  | **499.685**   | **0.98717**  |
| GBM 0.50  | 1579.571  | 547.138   | 0.97746  |
| GBM 0.05   | 2470.072  | 974.428   | 0.94488  |
| GBM 0.95   | 1435.852  | 818.770   | 0.98137  |

### Test Set Metrics

| Model | RMSE      | MAE       | R2       |
|-------|-----------|-----------|----------|
| RF    | 2726.884  | 1311.138  | 0.93259  |
| GBM 0.50   | **2566.822**  | **1230.770**  | **0.94027**  |
| GBM 0.05   | 3449.857  | 1715.0888 | 0.89211  |
| GBM 0.95   | 2879.736  | 1624.346  | 0.92482  |

Additionaly, there is a comparison of R2 between the default and optimized ML models, along with a plot depicting true versus predicted values on the test set (GBM), including prediction intervals.

<img src="https://iili.io/2tAyOnR.png" width="500" height="350"/>

*Comparing Default and Optimized Models on Test Set using R2 Metric (GBM)*

<br>

<img src="https://iili.io/2tApFzx.png" width="500" height="350"/>

## Bachelor’s Thesis and Presentation
Both the thesis and presentation are available as PDFs in this repository. Files:
- [Bachelor's Thesis (Serbian)](https://raw.githubusercontent.com/aleksa-radojicic/second_hand_car_price_prediction/refs/heads/main/Aleksa_Radojičić_20240165_Thesis.pdf)
- [Presentation (Serbian)](https://raw.githubusercontent.com/aleksa-radojicic/second_hand_car_price_prediction/refs/heads/main/Aleksa_Radojičić_20240165_Presentation.pdf)

## Acknowledgements
I really want to thank my mentors [Sandro Radovanović](https://rs.linkedin.com/in/sandroradovanovic) and [Andrija Petrović](https://rs.linkedin.com/in/andrija-petrovic-20299ba2) for their guidance and support throughout this project.