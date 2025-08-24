# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
请遵循公理设计、契约式设计、函数式编程、数据导向编程和奥卡姆剃刀原则

## Development Commands

### 激活conda环境
```bash
conda activate aqi-pre
```

### Data Processing
```bash
# Process raw air quality and meteorological data
python data_process_me.py
```

### Model Training
```bash
# Train any model by modifying config.py model_name parameter
python train.py

# Available models: ['RNN', 'GRU', 'LSTM', 'TCN', 'STCN']
# Edit config.py to change model type and hyperparameters
```

### Model Evaluation
```bash
# Evaluate trained model performance
python eval.py
```

### Configuration
- All model parameters are controlled through `config.py`
- Change `model_name` to switch between different model architectures
- Data paths, hyperparameters, and training settings are in `config.py`

## Project Structure

### Key Files
- `config.py`: Central configuration for all models and training parameters
- `models.py`: Model definitions
- `train.py`: Training loop with validation and model saving
- `eval.py`: Model evaluation with RMSE, MAE, and R² metrics
- `utils.py`: Data loading, preprocessing, and utility functions
- `data_process_me.py`: Raw data processing pipeline

### Data Organization
- `data/hezhou_air_data/`: Hezhou city air quality and meteorological data
- `data/microsoft_urban_air_data/`: Microsoft Research urban air dataset
- `data/stations_data/`: Individual Beijing monitoring station data
- `data/xy/`: Processed pickled data for model training
- `models/`: Saved model checkpoints

## Important Notes

- The project uses PyTorch and scikit-learn for deep learning
- Data is split temporally by month, not randomly
- Models are saved to `models/` directory with validation-based checkpointing
