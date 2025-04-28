```mermaid
graph TD
    subgraph "Satellite Data Sources"
        SENTINEL[Sentinel-5P]
        MODIS[MODIS]
        VIIRS[VIIRS]
        LANDSAT[Landsat]
    end

    subgraph "Google Earth Engine"
        GEE[GEE API]
        CATALOG[Data Catalog]
        PROCESS[Processing Engine]
    end

    subgraph "Data Collection & Processing"
        DC[Data Collector]
        SCHED[Scheduler]
        VALID[Data Validator]
        TRANS[Data Transformer]
        CLEAN[Data Cleaner]
        FEAT[Feature Extractor]
        AGG[Data Aggregator]
    end

    subgraph "Storage Layer"
        RAW[Raw Data Store]
        PROC[Processed Data Store]
        CACHE[Cache]
        MODELS[Model Storage]
    end

    subgraph "Machine Learning Pipeline"
        BM[Base Models]
        ML[Meta Learner]
        TS[Target Scaler]
        TRAIN[Model Trainer]
        EVAL[Model Evaluator]
    end

    subgraph "Prediction System"
        PREDICT[Predict Endpoint]
        BASE[Base Model Predictions]
        META[Meta Learner Prediction]
        FORECAST[Time Series Forecast]
    end

    subgraph "Visualization & UI"
        WEB[Web Interface]
        HM[Heatmap Generator]
        TSV[Time Series Visualizer]
    end

    %% Connections
    SENTINEL --> GEE
    MODIS --> GEE
    VIIRS --> GEE
    LANDSAT --> GEE

    GEE --> CATALOG
    CATALOG --> PROCESS
    PROCESS --> DC

    DC --> VALID
    VALID --> TRANS
    TRANS --> RAW

    SCHED --> DC
    SCHED --> VALID
    SCHED --> TRANS

    RAW --> CLEAN
    CLEAN --> FEAT
    FEAT --> AGG
    AGG --> PROC

    PROC --> CACHE
    PROC --> TRAIN
    TRAIN --> BM
    BM --> ML
    ML --> TS
    TS --> MODELS

    MODELS --> PREDICT
    PREDICT --> BASE
    BASE --> META
    META --> FORECAST
    FORECAST --> WEB

    WEB --> HM
    WEB --> TSV

    HM --> WEB
    TSV --> WEB

    %% Styling
    classDef satellite fill:#e8f5e9,stroke:#43a047,stroke-width:2px;
    classDef gee fill:#bbdefb,stroke:#1976d2,stroke-width:2px;
    classDef processor fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#fb8c00,stroke-width:2px;
    classDef ml fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef prediction fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef ui fill:#e8eaf6,stroke:#3949ab,stroke-width:2px;
    
    class SENTINEL,MODIS,VIIRS,LANDSAT satellite;
    class GEE,CATALOG,PROCESS gee;
    class DC,SCHED,VALID,TRANS,CLEAN,FEAT,AGG processor;
    class RAW,PROC,CACHE,MODELS storage;
    class BM,ML,TS,TRAIN,EVAL ml;
    class PREDICT,BASE,META,FORECAST prediction;
    class WEB,HM,TSV ui;
```

## Prediction System Description

### 1. Prediction Endpoint
- **Predict Endpoint**: Main prediction API
  - Handles city-based requests
  - Integrates weather and pollution data
  - Returns predictions and forecasts

### 2. Prediction Process
- **Base Model Predictions**
  - Multiple model predictions
  - Feature processing
  - Data validation
- **Meta Learner Prediction**
  - Combines base model outputs
  - Applies target scaling
  - Generates final prediction
- **Time Series Forecast**
  - Generates future predictions
  - Creates visualization data
  - Handles prediction bounds

### 3. Integration
- Direct connection to web interface
- Real-time prediction updates
- Interactive visualization
- Error handling and validation

## Prediction Workflow
1. User request through web interface
2. Data collection and validation
3. Base model predictions
4. Meta learner combination
5. Forecast generation
6. Visualization update 