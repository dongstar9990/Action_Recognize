Để xây dựng một mạng Convolutional Neural Network (CNN) cho việc phân loại hình ảnh, chúng ta có thể sử dụng Python và thư viện Keras của TensorFlow. Dưới đây là hướng dẫn từng bước để xây dựng một mô hình CNN cơ bản.

### 1. Cài đặt thư viện cần thiết:

```bash
pip install tensorflow
```

### 2. Nhập các thư viện cần thiết:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
```

### 3. Tải và chuẩn bị dữ liệu:

Chúng ta sẽ sử dụng bộ dữ liệu MNIST để phân loại hình ảnh chữ số viết tay.

```python
# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuyển đổi dữ liệu
x_train = np.expand_dims(x_train, axis=-1)  # Thêm chiều cho kênh màu
x_test = np.expand_dims(x_test, axis=-1)

# Tiền xử lý dữ liệu: chuẩn hóa
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

### 4. Xây dựng mô hình CNN:

```python
def build_cnn_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 lớp cho 10 chữ số
    return model
```

### 5. Biên soạn mô hình:

```python
model = build_cnn_model()
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

### 6. Huấn luyện mô hình:

```python
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
```

### 7. Đánh giá mô hình:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

### 8. Vẽ biểu đồ kết quả huấn luyện (Tuỳ chọn):

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
```

### Kết luận

Đoạn mã trên mô tả cách tạo ra một mô hình CNN cơ bản để phân loại hình ảnh trong bộ dữ liệu MNIST. Bạn có thể điều chỉnh kiến trúc của mô hình, số lượng lớp và các tham số khác để tối ưu hóa kết quả. Nếu bạn cần thêm thông tin chi tiết hoặc hướng dẫn về các phần cụ thể, hãy cho tôi biết!