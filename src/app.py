import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from vnstock import VnStock
from function import get_list_symbols_by_industry, get_list_industries
import io

# Tiêu đề ứng dụng
st.title("Application of Decision Trees in Predicting Stock Price Trends")

# Lấy danh sách ngành
industries = get_list_industries()
selected_industry = st.selectbox("Select Industry", options=industries)

# Lấy danh sách cổ phiếu theo ngành đã chọn
symbols = get_list_symbols_by_industry(selected_industry)
selected_stock = st.selectbox("Select Stock Symbol", options=symbols)

# Chọn ngày bắt đầu và ngày kết thúc
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", key="start_date", value="2014-01-01")
with col2:
    end_date = st.date_input("End Date", key="end_date")

# Chọn nguồn dữ liệu
data_source = st.selectbox(
    "Select Data Source",
    options=["VCI", "TCBS", "MSN", "HOSE"],
    index=0
)

# Thông báo khi sẵn sàng xử lý
if st.button("Submit"):
    st.write(f"## Analyzing {selected_stock} from {start_date} to {end_date}...")
    
    try:
        # Khởi tạo đối tượng phân tích
        vn_stock = VnStock()

        # Gọi phương thức analyze để thực hiện toàn bộ quy trình
        df, accuracy, cm = vn_stock.analyze(
            symbol=selected_stock.split(" - ")[0],  # Lấy mã cổ phiếu từ chuỗi "symbol - organ_name"
            source=data_source,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        # Hiển thị kết quả đánh giá
        st.subheader("Model Evaluation")
        st.markdown(f"<h4>Model Accuracy: <b style='color:green;'>{accuracy * 100:.2f}%</b></h4>", unsafe_allow_html=True)
        
        st.subheader("Confusion Matrix (Ma Trận Nhầm Lẫn)", divider="gray")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        st.pyplot(fig)
        
        # Thêm mô tả cho ma trận nhầm lẫn
        st.markdown("""
        ### Mô tả Ma Trận Confusion:
        - **True Negative (TN, 0-0):** Số lần mô hình dự đoán đúng xu hướng giảm.
        - **False Positive (FP, 0-1):** Số lần mô hình dự đoán sai xu hướng tăng trong khi thực tế là giảm.
        - **False Negative (FN, 1-0):** Số lần mô hình dự đoán sai xu hướng giảm trong khi thực tế là tăng.
        - **True Positive (TP, 1-1):** Số lần mô hình dự đoán đúng xu hướng tăng.
        
        **Ý nghĩa**: Ma trận nhầm lẫn giúp đánh giá chi tiết hiệu suất của mô hình, đặc biệt là khả năng phân loại đúng các trường hợp tăng/giảm giá cổ phiếu.
        """)

        # Hiển thị biểu đồ cây quyết định
        st.subheader("Decision Tree Visualization", divider="gray")
        fig = vn_stock.visualize_tree()
        st.pyplot(fig)
        
        # Mô tả cây quyết định
        st.markdown("### Mô tả Cây Quyết Định và Các Feature")
        st.markdown("""
        #### Cây Quyết Định:
        - Cây quyết định được sử dụng để dự đoán xu hướng giá cổ phiếu (Tăng hoặc Giảm).
        - Mỗi nút trong cây đại diện cho một điều kiện kiểm tra dựa trên các đặc trưng (features).
        - Nút lá chứa nhãn dự đoán cuối cùng (Tăng hoặc Giảm).

        #### Các Feature Được Sử Dụng:
        1. **open, high, low, close**: Dữ liệu giá cổ phiếu trong phiên giao dịch.
        2. **volume**: Khối lượng giao dịch, phản ánh sự quan tâm của thị trường.
        3. **MA_5, MA_20, MA_50**: Đường trung bình động, biểu thị xu hướng ngắn hạn và dài hạn.
        4. **RSI**: Chỉ số sức mạnh tương đối, xác định tình trạng mua quá mức hoặc bán quá mức.
        5. **bollinger_high, bollinger_low**: Biên trên và dưới của dải Bollinger, xác định ngưỡng hỗ trợ/kháng cự.
        6. **MACD, MACD_signal**: Chỉ báo động lượng và tín hiệu giao dịch.

        #### Ý Nghĩa:
        - Feature quan trọng nhất sẽ xuất hiện ở nút gốc và được sử dụng để phân chia dữ liệu.
        """)


        # Hiển thị biểu đồ độ quan trọng của đặc trưng
        st.subheader("Feature Importance", divider="gray")
        fig, ax = plt.subplots()
        vn_stock.plot_feature_importance()
        st.pyplot(fig)
        
        st.markdown("""
        ### Mô tả Feature Importance:
        - **Feature Importance** là một chỉ số cho biết mức độ quan trọng của từng đặc trưng (feature) trong việc đưa ra quyết định của mô hình.
        - Các giá trị cao hơn nghĩa là đặc trưng đó có ảnh hưởng lớn hơn đến kết quả dự đoán.

        #### Ý nghĩa của từng đặc trưng:
        - **Open, High, Low, Close**: Giá mở cửa, cao nhất, thấp nhất và đóng cửa của cổ phiếu trong phiên giao dịch.
        - **Volume**: Khối lượng giao dịch, biểu thị mức độ quan tâm của thị trường với cổ phiếu.
        - **MA (Moving Average)**: Đường trung bình động (5, 20, 50 ngày) biểu diễn xu hướng giá ngắn hạn và dài hạn.
        - **RSI (Relative Strength Index)**: Chỉ số sức mạnh tương đối, giúp xác định tình trạng mua quá mức (overbought) hoặc bán quá mức (oversold).
        - **Bollinger High / Low**: Biên trên và dưới của dải Bollinger, đo độ biến động giá.
        - **MACD, MACD Signal**: Chỉ báo động lượng và tín hiệu xu hướng giá.

        #### Sử dụng Feature Importance:
        - Đặc trưng có giá trị **Feature Importance** cao nhất là yếu tố chính ảnh hưởng đến dự đoán của mô hình.
        - Những đặc trưng quan trọng thấp có thể được loại bỏ để giảm độ phức tạp mà không ảnh hưởng đến hiệu suất.
        """)

        # Hiển thị dữ liệu đã xử lý
        st.subheader("Processed Data")
        st.dataframe(df)

        # Cung cấp nút tải xuống file Excel
        st.subheader("Download Processed Data")
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Download Processed Data",
            data=buffer,
            file_name=f"{selected_stock.split(' - ')[0]}_{start_date}_{end_date}.xlsx",
            mime="application/vnd.ms-excel"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
