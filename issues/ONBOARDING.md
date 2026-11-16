# Nhóm xác nhận clone và chạy test trên máy riêng

Owner: Hưng, Cầu, Tuấn. Mỗi bạn thực hiện trong phiên đầu tiên của mình.

Mục đích: xác nhận mỗi người đã có cùng source và môi trường chạy được trước
khi bắt đầu task riêng. Đây là kiểm tra môi trường, không phải benchmark.

1. Clone repo bằng tài khoản GitHub của mình và mở đúng thư mục có pyproject.toml.
2. Chạy các lệnh dưới trong PowerShell. Chưa cần tải encoder/LLM:

```powershell
git clone https://github.com/lamgiatran/SupportLens.git
cd SupportLens
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
git rev-parse --short HEAD
```

3. Nếu đã clone ở bước trước, bắt đầu từ `python -m venv .venv`, không clone lồng.
4. Comment trong issue: hệ điều hành, Python version, commit đang dùng, kết quả
   test và ảnh terminal. Nếu tokenizer thật chưa tải, ghi rõ skip và lý do.

- [ ] Hưng đã clone và gửi kết quả của máy mình.
- [ ] Cầu đã clone và gửi kết quả của máy mình.
- [ ] Tuấn đã clone và gửi kết quả của máy mình.

Xong khi: từng người tự chạy được hoặc đã gửi lỗi đầy đủ để Leader hỗ trợ.
Không đổi test hoặc tạo file giả chỉ để có cùng số passed với máy Leader.
Sau đó Cầu làm C-01/C-02, Hưng làm E1/H-01, Tuấn làm UI hai chế độ và mock stream.
