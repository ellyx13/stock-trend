from vnstock3 import Vnstock
stock = Vnstock().stock(symbol='FPT', source='VCI')
df = stock.quote.history(start='2024-01-01', end='2024-12-12', interval='1D')
# Lưu file Excel
# df.to_excel('transaction_history_fpt.xlsx', index=True) 
df['price_range'] = df['high'] - df['low']  # Biến động giá
df['close_change'] = (df['close'] - df['open']) / df['open']  # Tỷ lệ thay đổi giá
df['average_price'] = (df['high'] + df['low'] + df['close']) / 3  # Trung bình giá

# Gắn nhãn xu hướng
# 1: Giá tăng (giá đóng cửa hôm sau cao hơn hôm nay).
# 0: Giá giảm hoặc không đổi.
df['next_close'] = df['close'].shift(-1)
df['trend'] = (df['next_close'] > df['close']).astype(int)

# Xóa hàng cuối cùng vì không có giá đóng cửa ngày hôm sau
df = df.dropna()

print(df)

from sklearn.model_selection import train_test_split

# Biến đầu vào và biến mục tiêu
features = ['open', 'high', 'low', 'close', 'volume', 'price_range', 'close_change', 'average_price']
X = df[features]
y = df['trend']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Tạo mô hình cây quyết định
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá hiệu quả mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Hiển thị ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Trực quan hóa cây quyết định
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=features, class_names=['Giảm', 'Tăng'], filled=True)
plt.show()
