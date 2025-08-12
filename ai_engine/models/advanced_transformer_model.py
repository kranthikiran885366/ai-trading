import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets, seq_length=60):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class AdvancedFinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=6, d_ff=1024, 
                 max_seq_length=100, num_classes=3, dropout=0.1):
        super(AdvancedFinancialTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Risk prediction head
        self.risk_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)  # BULL, BEAR, SIDEWAYS, VOLATILE
        )
        
    def forward(self, x, mask=None):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Global pooling
        pooled = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Multiple outputs
        signal_logits = self.classifier(pooled)
        risk_score = self.risk_predictor(pooled)
        regime_logits = self.regime_classifier(pooled)
        
        return {
            'signal_logits': signal_logits,
            'risk_score': risk_score,
            'regime_logits': regime_logits,
            'features': pooled
        }

class FinBERTSentimentModel(nn.Module):
    def __init__(self, model_name='ProsusAI/finbert', num_classes=3, dropout=0.1):
        super(FinBERTSentimentModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Freeze BERT parameters for fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class QLoRAFineTuner:
    def __init__(self, model, rank=16, alpha=32, dropout=0.1):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers = {}
        
    def add_lora_layers(self):
        """Add LoRA layers to the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Add LoRA adaptation
                in_features = module.in_features
                out_features = module.out_features
                
                # LoRA matrices
                lora_A = nn.Parameter(torch.randn(in_features, self.rank) * 0.01)
                lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
                
                self.lora_layers[name] = {
                    'A': lora_A,
                    'B': lora_B,
                    'original': module
                }
                
                # Replace forward method
                def lora_forward(x, A=lora_A, B=lora_B, original=module):
                    original_output = original(x)
                    lora_output = torch.matmul(torch.matmul(x, A), B) * (self.alpha / self.rank)
                    return original_output + lora_output
                
                module.forward = lora_forward

class AdvancedModelTrainer:
    def __init__(self, input_dim, device=None):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.models = {}
        self.tokenizer = None
        
        # Initialize models
        self._initialize_models()
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self):
        """Initialize all models"""
        # Main transformer model
        self.models['transformer'] = AdvancedFinancialTransformer(
            input_dim=self.input_dim,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            num_classes=3,
            dropout=0.1
        ).to(self.device)
        
        # FinBERT for sentiment
        try:
            self.models['finbert'] = FinBERTSentimentModel().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        except Exception as e:
            self.logger.warning(f"Could not load FinBERT: {e}")
            self.models['finbert'] = None
        
        # QLoRA fine-tuner
        self.qlora_tuner = QLoRAFineTuner(self.models['transformer'])
        
    def prepare_sequences(self, data, target_col='signal', seq_length=60):
        """Prepare sequences for training"""
        features = data.drop([target_col], axis=1, errors='ignore')
        targets = data[target_col]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i])
            y.append(targets.iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_transformer(self, X, y, epochs=50, batch_size=32, learning_rate=1e-4):
        """Train the transformer model"""
        self.logger.info("Training transformer model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = FinancialTimeSeriesDataset(X_train, y_train)
        val_dataset = FinancialTimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer and loss
        model = self.models['transformer']
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        signal_criterion = nn.CrossEntropyLoss()
        risk_criterion = nn.MSELoss()
        regime_criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                
                # Multi-task loss
                signal_loss = signal_criterion(outputs['signal_logits'], batch_y)
                
                # Generate synthetic risk and regime targets for training
                risk_targets = torch.rand(batch_y.size(0), 1).to(self.device)
                regime_targets = torch.randint(0, 4, (batch_y.size(0),)).to(self.device)
                
                risk_loss = risk_criterion(outputs['risk_score'], risk_targets)
                regime_loss = regime_criterion(outputs['regime_logits'], regime_targets)
                
                total_loss = signal_loss + 0.3 * risk_loss + 0.2 * regime_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    
                    signal_loss = signal_criterion(outputs['signal_logits'], batch_y)
                    val_loss += signal_loss.item()
                    
                    _, predicted = torch.max(outputs['signal_logits'].data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            self.logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                           f'Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model('transformer', 'best_transformer_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
            
            scheduler.step()
        
        self.logger.info("Transformer training completed")
    
    def fine_tune_with_qlora(self, X, y, epochs=20, learning_rate=1e-5):
        """Fine-tune model using QLoRA"""
        self.logger.info("Fine-tuning with QLoRA...")
        
        # Add LoRA layers
        self.qlora_tuner.add_lora_layers()
        
        # Only train LoRA parameters
        lora_params = []
        for layer_info in self.qlora_tuner.lora_layers.values():
            lora_params.extend([layer_info['A'], layer_info['B']])
        
        optimizer = optim.AdamW(lora_params, lr=learning_rate)
        
        # Training loop (simplified)
        model = self.models['transformer']
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_tensor)
            loss = criterion(outputs['signal_logits'], y_tensor)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                self.logger.info(f'QLoRA Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        self.logger.info("QLoRA fine-tuning completed")
    
    def train_sentiment_model(self, texts, labels, epochs=10, batch_size=16):
        """Train FinBERT sentiment model"""
        if self.models['finbert'] is None or self.tokenizer is None:
            self.logger.warning("FinBERT model not available")
            return
        
        self.logger.info("Training FinBERT sentiment model...")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=512, return_tensors='pt'
        )
        
        # Create dataset
        class SentimentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        dataset = SentimentDataset(encodings, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = self.models['finbert']
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f'Sentiment Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        self.logger.info("FinBERT training completed")
    
    def predict(self, X, model_name='transformer'):
        """Make predictions"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = model(X_tensor)
            
            if model_name == 'transformer':
                signal_probs = torch.softmax(outputs['signal_logits'], dim=1)
                risk_scores = outputs['risk_score']
                regime_probs = torch.softmax(outputs['regime_logits'], dim=1)
                
                return {
                    'signal_probabilities': signal_probs.cpu().numpy(),
                    'risk_scores': risk_scores.cpu().numpy(),
                    'regime_probabilities': regime_probs.cpu().numpy(),
                    'features': outputs['features'].cpu().numpy()
                }
            else:
                return torch.softmax(outputs, dim=1).cpu().numpy()
    
    def predict_sentiment(self, texts):
        """Predict sentiment for texts"""
        if self.models['finbert'] is None or self.tokenizer is None:
            return None
        
        model = self.models['finbert']
        model.eval()
        
        results = []
        for text in texts:
            encoding = self.tokenizer(
                text, truncation=True, padding=True, max_length=512, return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                
                results.append(probabilities.cpu().numpy()[0])
        
        return np.array(results)
    
    def save_model(self, model_name, path):
        """Save model"""
        model = self.models[model_name]
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim
        }, path)
        
        self.logger.info(f"Model {model_name} saved to {path}")
    
    def load_model(self, model_name, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        model = self.models[model_name]
        model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        self.logger.info(f"Model {model_name} loaded from {path}")
    
    def get_model_summary(self):
        """Get model summary"""
        summary = {}
        for name, model in self.models.items():
            if model is not None:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                summary[name] = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
                }
        
        return summary

def main():
    """Test the advanced model trainer"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    seq_length = 60
    
    # Generate synthetic financial data
    data = pd.DataFrame(np.random.randn(n_samples, n_features))
    data['signal'] = np.random.randint(0, 3, n_samples)  # 0: SELL, 1: HOLD, 2: BUY
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(input_dim=n_features)
    
    # Prepare sequences
    X, y = trainer.prepare_sequences(data, seq_length=seq_length)
    
    print(f"Training data shape: {X.shape}")
    print(f"Model summary: {trainer.get_model_summary()}")
    
    # Train transformer
    trainer.train_transformer(X, y, epochs=5, batch_size=16)
    
    # Fine-tune with QLoRA
    trainer.fine_tune_with_qlora(X, y, epochs=5)
    
    # Make predictions
    predictions = trainer.predict(X[:10])
    print(f"Sample predictions: {predictions['signal_probabilities'][:5]}")

if __name__ == "__main__":
    main()
