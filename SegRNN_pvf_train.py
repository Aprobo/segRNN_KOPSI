import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 시드 고정
np.random.seed(42)
torch.manual_seed(42)

# 데이터 로드 및 일별 데이터로 변환
data = pd.read_csv('./data/kospi_pvf.csv')
data['Date'] = pd.to_datetime(data['Date']) #Dtype: object -> dateimte64
data.set_index('Date', inplace=True)
dates = data.index.to_numpy()
values = data['Price'].values.reshape(-1, 1)

feature_cols = ['Price','Volume','Fluct']
values = data[feature_cols]
values_price = data['Price'].values
# print(values.shape) #(6308,3)

      
# 훈련/검증/테스트 데이터 분할
train_size = int(len(values) * 0.6)
val_size = int(len(values) * 0.2)
test_size  = len(values) - train_size - val_size

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

train_base_idx = 0
val_base_idx   = train_size
test_base_idx  = train_size + val_size

# 데이터 정규화
scaler_P = MinMaxScaler()
scaler_V = MinMaxScaler()
scaler_F = MinMaxScaler()
scaler_Y = MinMaxScaler()

train_P = scaler_P.fit_transform(train_data[['Price']])
train_V = scaler_V.fit_transform(train_data[['Volume']])
train_F = scaler_F.fit_transform(train_data[['Fluct']])

val_P = scaler_P.transform(val_data[['Price']])
val_V = scaler_V.transform(val_data[['Volume']])
val_F = scaler_F.transform(val_data[['Fluct']])

test_P = scaler_P.transform(test_data[['Price']])
test_V = scaler_V.transform(test_data[['Volume']])
test_F = scaler_F.transform(test_data[['Fluct']])

train_scaled = np.concatenate([train_P, train_V, train_F], axis=1)   # (train, 3)
val_scaled  = np.concatenate([val_P,   val_V,   val_F],   axis=1)   # (val, 3)
test_scaled = np.concatenate([test_P,  test_V,  test_F],  axis=1)   # (test, 3)


# 세그먼트 분할 함수
def segment_data(data, segment_length):
    segments = []
    for start in range(0, len(data) - segment_length + 1, segment_length):
        segments.append(data[start:start + segment_length])
    return np.array(segments)

# 파라미터 설정
segment_length = 30  # 30일 단위 세그먼트로 설정
look_back = 7  # 각 세그먼트 내에서 예측에 사용할 이전 데이터의 수

# 세그먼트 기반 데이터셋 생성
train_segments = segment_data(train_scaled, segment_length) #(train_segments.shape) #[126,30,3] => 3780
val_segments = segment_data(val_scaled, segment_length)
test_segments = segment_data(test_scaled, segment_length)

# 데이터셋 생성 함수
def create_segmented_dataset(segments, base_start_idx, segment_length, look_back):
    X, Y, idxs = [], [], []
    for seg_idx, segment in enumerate(segments):
        seg_global_start = base_start_idx + seg_idx * segment_length

        for i in range(len(segment) - look_back):
            X.append(segment[i:(i + look_back)])
            next_step = segment[i + look_back]        # shape: (3,)
            Y.append(next_step[0])   # (look_back, 1)
            idxs.append(seg_global_start + i + look_back)

    return np.array(X), np.array(Y), np.array(idxs)

train_X, train_Y, train_idx = create_segmented_dataset(train_segments, base_start_idx=train_base_idx,
    segment_length=segment_length, look_back=look_back) #train_X.shape(3150,5,3) =126*25
val_X, val_Y, val_idx = create_segmented_dataset(val_segments, base_start_idx=val_base_idx,
    segment_length=segment_length, look_back=look_back)
test_X, test_Y, test_idx = create_segmented_dataset(test_segments, base_start_idx=test_base_idx,
    segment_length=segment_length, look_back=look_back)

# PyTorch Dataset 정의
class TimeSeriesSegmentDataset(Dataset):
    def __init__(self, X, Y, idxs):
        self.X = torch.tensor(X, dtype=torch.float32) #(N,7,3)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
        self.idxs = idxs 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.idxs[idx]

# DataLoader 생성
batch_size = 64
train_dataset = TimeSeriesSegmentDataset(train_X, train_Y, train_idx)
val_dataset = TimeSeriesSegmentDataset(val_X, val_Y, val_idx)
test_dataset = TimeSeriesSegmentDataset(test_X, test_Y,test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화
input_size = 3
hidden_size = 50
output_size = 1
num_layers = 1

# GRU 모델 정의
class GRUSegmentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUSegmentModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


model = GRUSegmentModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 모델 훈련 함수 수정 (Best loss 추적 및 저장)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    epoches_per_bar = 100
    total_bars = num_epochs // epoches_per_bar

    best_loss = float('inf')
    best_mse  = None
    best_model_weights = None

    for bar_idx in range(total_bars):
        print(f"\nProgress {bar_idx+1}/{total_bars}")

        with tqdm(total=epoches_per_bar, desc=f"Epochs {bar_idx*100+1}~{(bar_idx+1)*100}", unit="epoch") as pbar:

            for epoch_offset in range(epoches_per_bar):
                
                epoch = bar_idx * epoches_per_bar + epoch_offset
                model.train()
                train_loss = 0.0

                for inputs, targets, _ in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = inputs.view(inputs.size(0), look_back, -1)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, _ in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs = inputs.view(inputs.size(0), look_back, -1)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, targets).item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_weights = model.state_dict()

                    # Test MSE (정규화된 기준)
                    test_pred, test_act, _ = predict(model, test_loader)
                    best_mse = np.mean((test_act - test_pred)**2)

                # tqdm bar 1 epoch만큼 업데이트
                pbar.update(1)
    # 학습 끝난 후 best 가중치 저장
    torch.save({
        "state_dict": best_model_weights,
        "best_loss": best_loss,
        "best_mse": best_mse}, WEIGHT_PATH)

    return train_losses, val_losses, best_loss, best_mse

# 예측 함수
def predict(model, loader):
    model.eval()
    preds, acts, idxs = [], [], []

    with torch.no_grad():
        for inputs, targets, batch_idxs in loader:
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), look_back, -1)
            outputs = model(inputs.view(inputs.size(0), look_back, -1))

            preds.append(outputs.cpu().numpy().reshape(-1,1))  
            acts.append(targets.numpy().reshape(-1,1))         
            idxs.append(batch_idxs.numpy())

    preds = np.concatenate(preds)
    acts  = np.concatenate(acts)
    idxs  = np.concatenate(idxs)
    return preds, acts, idxs

WEIGHT_PATH = "best_model_weights.pth"

if __name__ == "__main__":
    train_losses, val_losses, best_loss, best_mse = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000)

    
    print(f"Best Validation Loss: {best_loss:.10f}")
    print(f"Best MSE on Test Set: {best_mse:.10f}")

    train_pred, train_act, train_idx = predict(model, train_loader)
    val_pred,   val_act,   val_idx   = predict(model, val_loader)
    test_pred,  test_act,  test_idx  = predict(model, test_loader)

    # 역정규화
    train_pred = scaler_P.inverse_transform(train_pred)
    val_pred   = scaler_P.inverse_transform(val_pred)
    test_pred  = scaler_P.inverse_transform(test_pred)
    test_act   = scaler_P.inverse_transform(test_act)

    train_pred = train_pred.reshape(-1)  
    val_pred   = val_pred.reshape(-1)
    test_pred  = test_pred.reshape(-1)
    test_act   = test_act.reshape(-1)    
   
    # 날짜 복원
    train_dates = dates[train_idx]
    val_dates   = dates[val_idx]
    test_dates  = dates[test_idx]

    # 테스트 데이터의 MSE 값 출력
    mse_value = np.mean((test_act - test_pred)**2)
    rmse_value = np.sqrt(mse_value)
    print(f"Test Set RMSE: {rmse_value:.4f}")

   # 마지막 몇 일의 결과 출력
    num_days_to_display = 5

    actual_last_few_days    = test_act[-num_days_to_display:]
    predicted_last_few_days = test_pred[-num_days_to_display:]
    dates_last              = test_dates[-num_days_to_display:]

    for date, actual, predicted in zip(dates_last, actual_last_few_days, predicted_last_few_days):
        date = pd.to_datetime(date)
        print(f"Date: {date.strftime('%Y-%m-%d')}, Actual: {actual:.2f}, Predicted: {predicted:.2f}")

    # 예측 결과 시각화
    plt.figure(figsize=(16,8))

    # 실제 KOSPI
    plt.plot(dates, values_price, label='Actual KOSPI', color='black')

    # 시각화
    plt.plot(train_dates, train_pred.reshape(-1), label='Train Prediction', color='red',   alpha=0.6)
    plt.plot(val_dates,   val_pred.reshape(-1),   label='Val Prediction',   color='blue',  alpha=0.6)
    plt.plot(test_dates,  test_pred.reshape(-1),  label='Test Prediction',  color='green', alpha=0.6)
    plt.title('KOSPI Daily Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('KOSPI')
    plt.legend()
    plt.savefig('./result/pvf/prediction.png')  # 그래프를 이미지로 저장
    plt.show()
  
    #test
    plt.figure(figsize=(16,8))
    plt.plot(test_dates, test_act, label='Actual Test KOSPI', color='black')
    # 테스트 구간 예측값
    plt.plot(test_dates, test_pred, label='Predicted Test KOSPI', color='green', alpha=0.6)

    plt.title('KOSPI Test Period Prediction')
    plt.xlabel('Date')
    plt.ylabel('KOSPI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('./result/pvf/test_prediction.png')
    plt.show()

    # 손실 함수 추이 그래프 시각화 수정
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(16, 8))  # 그래프 크기 키움
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.title('Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.yscale('log')  # y축을 로그 스케일로 변환
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('./result/pvf/loss_over_epochs_log_scale.png')  # 그래프를 이미지로 저장
    plt.show()