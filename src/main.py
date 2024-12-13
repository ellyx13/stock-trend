import matplotlib.pyplot as plt
import ta
from vnstock3 import Vnstock
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    df = stock.quote.history(start=start_date, end=end_date, interval='1D')
    return df

def calculate_technical_indicators(df):
    df['price_range'] = df['high'] - df['low']
    df['close_change'] = (df['close'] - df['open']) / df['open']
    df['average_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    macd = ta.trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['Ratio_Close_MA50'] = df['close'] / df['MA_50']
    df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    df.dropna(subset=['Ratio_Close_MA50'], inplace=True)
    return df

def label_trends(df):
    df['next_close'] = df['close'].shift(-1)
    df['trend'] = (df['next_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def prepare_features(df):
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'MA_5', 'MA_20', 'MA_50',
        'RSI', 'bollinger_high', 'bollinger_low',
        'MACD', 'MACD_signal'
    ]
    X = df[features]
    y = df['trend']
    return X, y, features

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return y_pred

def visualize_tree(model, features):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=features, class_names=['Decrease', 'Increase'], filled=True)
    plt.show()

def cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", scores)

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.show()
    
def export_to_excel(df, symbol, start_date, end_date):
    filename = f"{symbol}_{start_date}_{end_date}.xlsx"
    file_path = "data/" + filename
    df.to_excel(file_path, index=False)
    print(f"Data exported to {filename}")
    
def main(symbol='FPT', start_date='2014-01-01', end_date='2023-10-18'):
    df = fetch_stock_data(symbol, start_date, end_date)
    df = calculate_technical_indicators(df)
    df = label_trends(df)
    export_to_excel(df, symbol, start_date, end_date)
    X, y, features = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(X, y)
    model = build_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    visualize_tree(model, features)
    cross_validation(model, X, y)
    plot_feature_importance(model, features)

main()