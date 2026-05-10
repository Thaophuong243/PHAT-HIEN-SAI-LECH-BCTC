import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống Chẩn đoán Gian lận Tài chính", layout="wide")

class FraudDetectionApp:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.features = [
            'Total_Assets', 'Total_Liabilities', 'Revenue', 'Operating_Expenses', 
            'Net_Income', 'Cash_Flow_Operating', 'Cash_Flow_Investing', 
            'Cash_Flow_Financing', 'Current_Ratio', 'Debt_to_Equity', 
            'Gross_Margin', 'Return_on_Assets', 'Return_on_Equity'
        ]

    @st.cache_resource
    def train_model(_self):
        """Huấn luyện mô hình mỗi khi ứng dụng khởi chạy"""
        try:
            df = pd.read_csv(_self.data_path)
            X = df[_self.features]
            y = df['Financial_Status']
            # Tính toán trọng số chính xác hoặc gán các hình phạt lệch mạnh
            custom_weights = {0: 1, 1: 10, 2: 50} 
            
            rf = RandomForestClassifier(
                n_estimators=200, 
                class_weight=custom_weights, 
                max_depth=8,              # Ràng buộc độ sâu để ngăn ngừa quá khớp
                min_samples_leaf=5,       # Yêu cầu các nút lá mang tính tổng quát
                random_state=42,
                n_jobs=-1
            )
            # Chỉ huấn luyện mô hình trên dữ liệu huấn luyện đã được phân tách
            rf.fit(X, y)
            return rf, df
        except Exception as e:
            st.error(f"Lỗi khi huấn luyện mô hình: {e}")
            return None, None

    def run(self):
        st.title("🛡️ Hệ thống Phát hiện Gian lận Báo cáo Tài chính")
        st.markdown("---")

        # Khởi tạo mô hình
        with st.spinner("Đang huấn luyện mô hình thực tế..."):
            self.model, full_data = self.train_model()

        if self.model is None:
            return

        # Giao diện chính chia làm 2 cột
        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("📋 Nhập dữ liệu tài chính")
            input_data = {}

            # Tạo các ô nhập liệu cho 13 chỉ số
            for feature in self.features:
                val_str = st.text_input(f"{feature}", value="0.0")
                
                # Ép kiểu từ chữ sang số thực (float)
                try:
                    input_data[feature] = float(val_str)
                except ValueError:
                    st.error(f"⚠️ Vui lòng chỉ nhập số cho trường {feature}!")
                    input_data[feature] = 0.0
            
            predict_btn = st.button("🔍 Thực hiện chẩn đoán", use_container_width=True)

        with col2:
            st.header("📊 Kết quả dự báo")
            if predict_btn:
                # Chuyển dữ liệu nhập vào thành DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                prediction = self.model.predict(input_df)[0]
                probability = self.model.predict_proba(input_df)

                # Hiển thị kết quả trực quan
                if prediction == 0:
                    st.success("✅ Kết quả: Báo cáo Tài chính Bình thường")
                elif prediction == 1:
                    st.warning("⚠️ Kết quả: Có dấu hiệu Gian lận (Loại 1)")
                else:
                    st.error("🚨 Kết quả: Nghi vấn Gian lận Nghiêm trọng (Loại 2)")

                # Hiển thị xác suất
                st.write("### Xác suất chi tiết:")
                prob_df = pd.DataFrame(
                    probability, 
                    columns=["Bình thường (0)", "Nghi vấn (1)", "Rủi ro cao (2)"]
                )
                st.dataframe(prob_df.style.highlight_max(axis=1))

            else:
                st.info("Vui lòng nhập các thông số bên trái và ấn nút Chẩn đoán.")


if __name__ == "__main__":
    # Đảm bảo file train_data.csv nằm cùng thư mục với app.py
    app = FraudDetectionApp("train_data.csv")
    app.run()