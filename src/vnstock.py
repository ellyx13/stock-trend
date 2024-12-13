import matplotlib.pyplot as plt
import ta
from vnstock3 import Vnstock
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


class VnStock:
    def __init__(self):
        self.vn_stock = Vnstock()
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'MA_5', 'MA_20', 'MA_50',
            'RSI', 'bollinger_high', 'bollinger_low',
            'MACD', 'MACD_signal'
        ]
        self.model = None

    def fetch_stock_data(self, symbol, source, start_date, end_date):
        """Fetch historical stock data."""
        try:
            stock = self.vn_stock.stock(symbol=symbol, source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            return df
        except Exception as e:
            raise RuntimeError(f"Error fetching stock data: {e}")


    def calculate_technical_indicators(self, df):
        """Add technical indicators to the DataFrame."""
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
        df.dropna(inplace=True)
        return df


    def label_trends(self, df):
        """Label data with trends."""
        df['next_close'] = df['close'].shift(-1)
        df['trend'] = (df['next_close'] > df['close']).astype(int)
        df.dropna(inplace=True)
        return df


    def prepare_features(self, df):
        """Prepare features and target for modeling."""
        x = df[self.features]
        y = df['trend']
        return x, y


    def split_data(self, x, y):
        """Split data into training and testing sets."""
        return train_test_split(x, y, test_size=0.2, random_state=42)


    def build_model(self, x_train, y_train):
        """Build and tune a Decision Tree model."""
        param_grid = {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        return self.model

    def evaluate_model(self, x_test, y_test):
        """Evaluate model performance."""
        if not self.model:
            raise RuntimeError("Model has not been built yet.")
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, cm


    def visualize_tree(self):
        """Visualize the Decision Tree."""
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(self.model, feature_names=self.features, class_names=['Giảm', 'Tăng'], filled=True, ax=ax)
        return fig

    def cross_validation(self, model, x, y):
        scores = cross_val_score(model, x, y, cv=5)
        print("Cross-validation scores:", scores)

    def plot_feature_importance(self):
        """Plot feature importance."""
        importances = self.model.feature_importances_
        plt.barh(self.features, importances)
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance")
        plt.show()

    def export_to_excel(self, df, symbol, start_date, end_date):
        filename = f"{symbol}_{start_date}_{end_date}.xlsx"
        file_path = "data/" + filename
        df.to_excel(file_path, index=False)
        print(f"Data exported to {filename}")

    def analyze(self, symbol, source, start_date, end_date, show=True):
        """Complete analysis pipeline."""
        print("Fetching stock data...")
        df = self.fetch_stock_data(symbol, source, start_date, end_date)
        print("Calculating technical indicators...")
        df = self.calculate_technical_indicators(df)
        print("Labeling trends...")
        df = self.label_trends(df)
        df.reset_index(drop=True, inplace=True)  # Reset chỉ số
        print("Preparing features...")
        x, y = self.prepare_features(df)
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        print("Building model...")
        self.build_model(x_train, y_train)
        print("Evaluating model...")
        accuracy, cm = self.evaluate_model(x_test, y_test)
        print(f"Model Accuracy: {accuracy:.2f}")
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        if show: 
            plt.show()
            print("Visualizing Decision Tree...")
            self.visualize_tree()
            print("Plotting Feature Importance...")
            self.plot_feature_importance()
        return df, accuracy, cm