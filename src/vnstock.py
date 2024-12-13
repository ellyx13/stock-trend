import matplotlib.pyplot as plt
import ta
from vnstock3 import Vnstock
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import pandas as pd
import streamlit as st
from prophet.plot import plot_plotly
from prophet import Prophet
import plotly.graph_objects as go

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
            'max_depth': [3, 5, 10, 15, None],  # Thêm giá trị None để kiểm tra cây không giới hạn độ sâu
            'min_samples_split': [2, 5, 10],  # Số lượng mẫu tối thiểu để chia một nút
            'min_samples_leaf': [1, 2, 5],  # Số lượng mẫu tối thiểu để ở lại một lá
            'criterion': ['gini', 'entropy'],  # Tiêu chí để đo độ tinh khiết của nút
            'splitter': ['best', 'random']  # Kiểm tra cả cách chia tốt nhất và chia ngẫu nhiên
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
        self.y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, self.y_pred)
        cm = confusion_matrix(y_test, self.y_pred)
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
        
    def calculate_precision(self):
        return precision_score(self.y_test, self.y_pred)

    def calculate_recall(self):
        return recall_score(self.y_test, self.y_pred)
    
    def calculate_f1_score(self):
        return f1_score(self.y_test, self.y_pred)

    def analyze(self, symbol, source, start_date, end_date):
        """Complete analysis pipeline."""
        print("Fetching stock data...")
        df = self.fetch_stock_data(symbol, source, start_date, end_date)
        print("Calculating technical indicators...")
        df = self.calculate_technical_indicators(df)
        print("Labeling trends...")
        self.df = self.label_trends(df)
        self.df.reset_index(drop=True, inplace=True)  # Reset chỉ số
        print("Preparing features...")
        x, y = self.prepare_features(df)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(x, y)
        print("Building model...")
        self.build_model(self.x_train, self.y_train)
        print("Evaluating model...")
        accuracy, cm = self.evaluate_model(self.x_test, self.y_test)
        print(f"Model Accuracy: {accuracy:.2f}")
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()
        print("Visualizing Decision Tree...")
        self.visualize_tree()
        print("Plotting Feature Importance...")
        self.plot_feature_importance()
        return self.df, accuracy, cm
    
    def forecast_tomorrow(self):
        future_data = self.df.iloc[-1:]  # Lấy dữ liệu của ngày gần nhất
        future_prediction = self.model.predict(future_data[self.features])
        if future_prediction[0] == 1:
            return True # Tăng
        return False # Giảm
    
    def _prepare_data_to_forecast(self):
        # Đổi tên cột 'time' thành 'ds' và 'close' thành 'y' theo yêu cầu của Prophet
        df = self.df.rename(columns={'time': 'ds', 'close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])  # Chuyển cột 'ds' thành định dạng datetime
        df['y'] = df['y'].astype(float)     # Đảm bảo cột 'y' là kiểu số thực
        return df[['ds', 'y']]  # Chỉ giữ lại hai cột cần thiết cho Prophet


    def plot_prophet_forecast_with_plotly(self):
        df = self._prepare_data_to_forecast()
        # Huấn luyện mô hình Prophet
        model = Prophet()
        model.fit(df)

        # Tạo dự báo cho 365 ngày tiếp theo
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Tạo biểu đồ Plotly
        fig = go.Figure()

        # Dữ liệu gốc
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Dữ liệu gốc'))

        # Dự báo
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Dự báo'))

        # Vùng tin cậy
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Vùng tin cậy trên',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Vùng tin cậy dưới',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,200,0.2)',  # Màu vùng tin cậy
            showlegend=True
        ))

        # Tùy chỉnh biểu đồ
        fig.update_layout(
            title="Dự báo giá cổ phiếu",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị (Close)",
            template="plotly_dark"
        )

        return fig



vn_stock = Vnstock()

@st.cache_data
def get_list_industries():
    industries = vn_stock.stock().listing.symbols_by_industries()["icb_name2"].unique().tolist()
    industries.insert(0, "Tất cả")
    return industries

@st.cache_data
def get_list_symbols_by_industry(industry):
    df = vn_stock.stock().listing.symbols_by_industries()
    if industry == "Tất cả":
        filtered_df = df
    else:
        filtered_df = df[df["icb_name2"] == industry]
    return (filtered_df["symbol"] + " - " + filtered_df["organ_name"]).tolist()