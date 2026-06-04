import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống Chẩn đoán Gian lận Tài chính", layout="wide", page_icon="🛡️")

# --- HÀM HUẤN LUYỆN MODEL (Được Cache để chạy nhanh) ---
@st.cache_resource
def train_model(data_path, features):
    try:
        df = pd.read_csv(data_path)
        X = df[features]
        y = df['Financial_Status']
        class_counts = y.value_counts()

        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            max_depth=12,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        return rf, class_counts
    except Exception as e:
        return None, str(e)

class FraudDetectionApp:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.features = [
            'Total_Assets', 'Total_Liabilities', 'Revenue', 'Operating_Expenses', 
            'Net_Income', 'Cash_Flow_Operating', 'Cash_Flow_Investing', 
            'Cash_Flow_Financing', 'Current_Ratio', 'Debt_to_Equity', 
            'Gross_Margin', 'Return_on_Assets', 'Return_on_Equity'
        ]
        self.model = None

    def run(self):
        st.title("🛡️ Hệ thống Phát hiện Gian lận Báo cáo Tài chính")
        st.markdown("Hỗ trợ kiểm toán viên phân tích rủi ro dựa trên dữ liệu học máy (Random Forest).")
        st.divider()

        # 1. Huấn luyện mô hình ngầm
        with st.spinner("Đang khởi tạo lõi phân tích AI..."):
            self.model, class_counts = train_model(self.data_path, self.features)

        if self.model is None:
            st.error(f"❌ Lỗi khởi tạo mô hình: {class_counts}. Vui lòng kiểm tra lại file {self.data_path}")
            return

        # 2. Sidebar - Thông tin mô hình
        with st.sidebar:
            st.header("📈 Thông tin mô hình huấn luyện")
            st.success("Trạng thái: Đã sẵn sàng")
            st.write("**Phân phối class gốc:**")
            dist_df = pd.DataFrame({
                'Trạng thái': ['Bình thường (0)', 'Nghi vấn (1)', 'Rủi ro cao (2)'],
                'Số lượng hồ sơ': [class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0)],
            })
            st.dataframe(dist_df, hide_index=True)
            st.caption("Thuật toán: Random Forest | Tối ưu: class_weight='balanced'")

        # 3. Giao diện chính - Khu vực tải file
        st.header("📂 Tải lên dữ liệu tài chính")
        uploaded_file = st.file_uploader("Kéo/Thả file CSV hoặc XLSX hoặc chọn từ máy tính", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Đọc file dựa trên định dạng
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Không thể đọc file. Lỗi: {e}")
                return

            # Kiểm tra xem file có đủ 13 cột đặc trưng không
            missing_features = [col for col in self.features if col not in user_df.columns]
            if missing_features:
                st.error(f"🚨 File tải lên bị thiếu các cột bắt buộc sau để phân tích: {', '.join(missing_features)}")
                st.info("Vui lòng tải lên file có cấu trúc cột giống với file huấn luyện ban đầu.")
                return

            st.success(f"Đã tải thành công file **{uploaded_file.name}** với {len(user_df)} dòng dữ liệu.")
            st.divider()

            # 4. Khu vực chọn dòng để phân tích
            st.header("🏢 Chọn công ty để phân tích")
            
            # Tạo danh sách options dạng "Dòng 1", "Dòng 2"...
            # Nếu file của bạn có cột 'Ten_Cong_Ty' hoặc 'Company', có thể thay đổi logic ở đây
            row_options = [f"Dòng {i+1}" for i in range(len(user_df))]
            selected_row_label = st.selectbox("Chọn dòng dữ liệu:", row_options)
            
            # Lấy index thực tế (Dòng 1 -> index 0)
            selected_index = int(selected_row_label.split(" ")[1]) - 1
            selected_data = user_df.iloc[[selected_index]] # Lấy DataFrame chứa 1 dòng
            
            # Hiển thị dữ liệu của dòng đang chọn
            st.write("📊 **Chi tiết các chỉ số của dòng đang chọn:**")
            st.dataframe(selected_data[self.features], hide_index=True)

            # 5. Khu vực Dự đoán
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_btn = st.button("🔍 TIẾN HÀNH CHẨN ĐOÁN RỦI RO", type="primary", use_container_width=True)

            if predict_btn:
                st.divider()
                st.header("📊 Kết quả dự báo")
                
                # Trích xuất đúng 13 features để đưa vào model
                input_df = selected_data[self.features]
                
                # Xử lý nếu có giá trị NaN (trống) trong file Excel
                if input_df.isnull().values.any():
                    st.warning("⚠️ Dữ liệu có ô trống. Hệ thống sẽ tự động điền giá trị 0 để tiếp tục.")
                    input_df = input_df.fillna(0)

                # Dự đoán
                prediction = self.model.predict(input_df)[0]
                probability = self.model.predict_proba(input_df)[0] # Lấy mảng 1 chiều

                # Thiết kế hiển thị kết quả
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.write("### Kết luận hệ thống:")
                    if prediction == 0:
                        st.success("✅ **BÁO CÁO BÌNH THƯỜNG**\n\nKhông phát hiện dấu hiệu bất thường trong các chỉ số tài chính.")
                    elif prediction == 1:
                        st.warning("⚠️ **CÓ DẤU HIỆU NGHI VẤN (LOẠI 1)**\n\nCần kiểm tra kỹ dòng tiền và tỷ lệ nợ.")
                    else:
                        st.error("🚨 **RỦI RO GIAN LẬN CAO (LOẠI 2)**\n\nPhát hiện nhiều mâu thuẫn nghiêm trọng trong cơ cấu tài chính. Yêu cầu kiểm toán khẩn cấp!")

                with res_col2:
                    st.write("### Xác suất chi tiết:")
                    # Dùng thanh tiến trình (progress bar) của Streamlit cho trực quan
                    st.write(f"🟢 Bình thường: **{probability[0]*100:.1f}%**")
                    st.progress(float(probability[0]))
                    
                    st.write(f"🟡 Nghi vấn: **{probability[1]*100:.1f}%**")
                    st.progress(float(probability[1]))
                    
                    st.write(f"🔴 Rủi ro cao: **{probability[2]*100:.1f}%**")
                    st.progress(float(probability[2]))

if __name__ == "__main__":
    app = FraudDetectionApp("train_data.csv")
    app.run()