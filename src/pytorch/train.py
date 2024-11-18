import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import gensim
from model import CNNText
import sys

# GoogleNews pre-trained vector loading
def load_pretrained_vectors(word_idx_map, embed_dim=300, binary=True, path="/hdd/user4/cnn/data/GoogleNews-vectors-negative300.bin"):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    vocab_size = len(word_idx_map) + 1
    pretrained_embed = np.zeros((vocab_size, embed_dim))

    for word, idx in word_idx_map.items():
        if word in word_vectors:
            pretrained_embed[idx] = word_vectors[word]
        else:
            pretrained_embed[idx] = np.random.uniform(-0.25, 0.25, size=(embed_dim,))

    return pretrained_embed

# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, data, word_idx_map, max_l):
        self.data = data
        self.word_idx_map = word_idx_map
        self.max_l = max_l

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text'].split()
        x = [self.word_idx_map.get(w, 0) for w in text]
        if len(x) < self.max_l:
            x += [0] * (self.max_l - len(x))
        else:
            x = x[:self.max_l]
        x = torch.tensor(x, dtype=torch.long)
        y = sample['y']
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# 학습 함수 정의
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc="학습 진행"):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 평가 함수 정의
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    corrects = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="평가 진행"):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            corrects += (preds == y).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = corrects / len(data_loader.dataset)
    return avg_loss, accuracy

# 10-fold 교차 검증 함수 정의
def cross_validate(model_type, dataset_name, folds=10):
    # 데이터 로드
    if dataset_name == "MR":
        with open('/hdd/user4/cnn/data/preprocessed/mr.bin', 'rb') as f:
            data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
    elif dataset_name == "CR":
        with open('/hdd/user4/cnn/data/preprocessed/cr.bin', 'rb') as f:
            data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
    elif dataset_name == "Subj":
        with open('/hdd/user4/cnn/data/preprocessed/subj.bin', 'rb') as f:
            data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
    elif dataset_name == "MPQA":
        with open('/hdd/user4/cnn/data/preprocessed/mpqa.bin', 'rb') as f:
            data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
    elif dataset_name == "TREC":
        with open('/hdd/user4/cnn/data/preprocessed/trec_train.bin', 'rb') as f:
            train_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
        with open('/hdd/user4/cnn/data/preprocessed/trec_test.bin', 'rb') as f:
            test_data, _, _, _, _, _ = pickle.load(f)
    elif dataset_name == "SST1":
        with open('/hdd/user4/cnn/data/preprocessed/sst1_train.bin', 'rb') as f:
            train_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
        with open('/hdd/user4/cnn/data/preprocessed/sst1_test.bin', 'rb') as f:
            test_data, _, _, _, _, _ = pickle.load(f)
        with open('/hdd/user4/cnn/data/preprocessed/sst1_dev.bin', 'rb') as f:
            dev_data, _, _, _, _, _ = pickle.load(f)
    elif dataset_name == "SST2":
        with open('/hdd/user4/cnn/data/preprocessed/sst2_train.bin', 'rb') as f:
            train_data, W, W2, word_idx_map, vocab, max_l = pickle.load(f)
        with open('/hdd/user4/cnn/data/preprocessed/sst2_test.bin', 'rb') as f:
            test_data, _, _, _, _, _ = pickle.load(f)
        with open('/hdd/user4/cnn/data/preprocessed/sst2_dev.bin', 'rb') as f:
            dev_data, _, _, _, _, _ = pickle.load(f)
    else:
        print("지원되지 않는 데이터셋입니다.")
        sys.exit()

    # 임베딩 준비
    if model_type == 'rand':
        embedding_dim = 300
        vocab_size = len(vocab) + 1
        pretrained_embed = None
    else:
        pretrained_embed = load_pretrained_vectors(word_idx_map)

    kernel_sizes = [3, 4, 5]
    num_channels = 100
    class_num = max([d['y'] for d in (train_data if dataset_name in ["TREC", "SST1", "SST2"] else data)]) + 1
    best_test_acc = 0
    best_val_acc = 0
    
    if dataset_name in ['TREC', 'SST1', 'SST2']:
        # TREC 데이터셋의 경우 별도로 학습/테스트 데이터를 분리
        model = CNNText(
            vocab_size=len(vocab) + 1,
            embed_dim=300,
            class_num=class_num,
            kernel_sizes=kernel_sizes,
            num_channels=num_channels,
            pretrained_embed=pretrained_embed if model_type != 'rand' else None,
            static=(model_type == 'static'),
            multichannel=(model_type == 'multichannel')
        )
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = TextDataset(train_data, word_idx_map, max_l)
        test_dataset = TextDataset(test_data, word_idx_map, max_l)
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=50)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adadelta(parameters, lr=0.7, rho=0.3, weight_decay=1e-3)
        early_stopping = EarlyStopping(patience=3, min_delta=0.0001)

        # 학습 시작
        num_epochs = 25
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - 훈련 손실: {train_loss:.4f} - 테스트 손실: {test_loss:.4f} - 테스트 정확도: {test_acc:.4f}")

            best_test_acc = max(best_test_acc, test_acc)
            
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            print(f"10-Fold 교차 검증 평균 정확도: {best_test_acc:.4f}")
    else:
        # KFold 객체 생성
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
            print(f"Fold {fold + 1}/{folds}")
            
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            # 모델 생성
            model = CNNText(
                vocab_size=len(vocab) + 1,
                embed_dim=300,
                class_num=class_num,
                kernel_sizes=kernel_sizes,
                num_channels=num_channels,
                pretrained_embed=pretrained_embed if model_type != 'rand' else None,
                static=(model_type == 'static'),
                multichannel=(model_type == 'multichannel')
            )
            
            # 데이터셋 및 데이터로더 생성
            train_dataset = TextDataset(train_data, word_idx_map, max_l)
            val_dataset = TextDataset(val_data, word_idx_map, max_l)
            train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=50)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adadelta(parameters, lr=1.0, rho=0.8, weight_decay=1e-8)
            early_stopping = EarlyStopping(patience=3, min_delta=0.0001)

            # 학습 시작
            num_epochs = 25
            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f"Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs} - 훈련 손실: {train_loss:.4f} - 검증 손실: {val_loss:.4f} - 검증 정확도: {val_acc:.4f}")
                
                # 최고 정확도 갱신
                best_val_acc = max(best_val_acc, val_acc)
                
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1} for fold {fold + 1}")
                    break

            fold_accuracies.append(val_acc)
        
        # 전체 fold 정확도 출력
        avg_accuracy = np.mean(fold_accuracies)
        print(f"10-Fold 교차 검증 평균 정확도: {avg_accuracy:.4f}")

# 메인 함수
if __name__ == "__main__":
    model_type = sys.argv[1]  # 'rand', 'static', 'non-static', 'multichannel'
    dataset_name = sys.argv[2]  # 'MR', 'CR', 'Subj', 'MPQA'
    cross_validate(model_type, dataset_name)