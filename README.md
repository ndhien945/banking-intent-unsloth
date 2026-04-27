# Banking Intent Classification with Unsloth (LLaMA 3)

Dự án này thực hiện việc tinh chỉnh (fine-tuning) mô hình ngôn ngữ lớn LLaMA 3 (8B) để phân loại ý định (intent classification) trong lĩnh vực ngân hàng sử dụng tập dữ liệu **BANKING77** và thư viện **Unsloth** để tối ưu hóa hiệu năng và bộ nhớ.

## 1. Yêu cầu hệ thống & Cài đặt môi trường

Dự án yêu cầu Python và môi trường có hỗ trợ GPU (CUDA) để tận dụng Unsloth.

### Cài đặt các thư viện cần thiết
Sử dụng tệp `requirements.txt` để cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```
### Cài đặt Unsloth
Unsloth có thể được cài đặt từ PyPI:
```bash
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Clone code về từ Github để xử lí trên môi trường Kaggle:
```bash
!git clone https://github.com/ndhien945/banking-intent-unsloth.git
%cd banking-intent-unsloth
!ls -la
```

## 2. Quy trình thực hiện

### Chuẩn bị và tiền xử lí dữ liệu

Sau khi tải tập dữ liệu BANKING77 từ Hugging Face và lưu vào thư mục `sample_data`, ta thực hiện lấy mẫu (sampling) và ánh xạ nhãn để phù hợp với tài nguyên tính toán
```bash
python scripts/preprocess_data.py --num_train_samples 5390 --num_test_samples 1540
```
Trong đó `num_train_samples` và `num_test_samples` có thể được điều chỉnh để giảm kích thước tập dữ liệu, giúp quá trình huấn luyện nhanh hơn và tiết kiệm bộ nhớ.

### Huấn luyện mô hình với Unsloth

Ta sẽ sử dụng kỹ thuật LoRA để tinh chỉnh mô hình LLaMA-3-8B 4-bit.
```bash
python scripts/train.py
```
Mô hình sau khi huấn luyện sẽ được lưu tại đường dẫn đã cấu hình trong `configs/train.yaml`, ở đây là `outputs/banking-intent-lora`

### Kết quả thực hiện

#### Kết quả trên từng câu đơn lẻ
```bash
python scripts/inference.py --message "Help! I just lost my wallet and my credit card is in it. What should I do?"
```

#### Kết quả trên toàn bộ tập test
```bash
python scripts/inference.py --evaluate sample_data/test.csv
```

## 3. Cấu hình & Siêu tham số (Hyperparameters)

| Parameter | Value |
|:-----------|:-------|
| Base Model | unsloth/llama-3-8b-bnb-4bit |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Batch Size | 2 (Gradient Accumulation Steps: 4) |
| Epochs | 1 |
| Optimizer | AdamW 8-bit |
| Max Sequence Length | 256 |
| Target modules | "q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "up_proj", "down_proj" |

### 4. Kết quả đánh giá
| Evaluation Metric | Value |
|:------------------|:------|
| Accuracy          | 87.92% |
| F1 Score          | 0.78904 |

### 5. Link demo

- Demo video: [GDrive](https://drive.google.com/file/d/1mu-asGyGbaXNR8uj2sLn0eiWSNRaEGG-/view?usp=sharing)

### 6. Link Kaggle cho phần Training và Inference

- Kaggle Notebook: [Banking77](https://www.kaggle.com/code/danghiennguyen/banking77)

