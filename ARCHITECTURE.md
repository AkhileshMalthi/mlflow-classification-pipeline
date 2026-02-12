# System Architecture

This document provides a high-level overview of the ML Classification Model system architecture, detailing component interactions, data flow, and deployment structure.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Development Environment                          │
│                                                                          │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────┐ │
│  │ Data Processor   │      │ Model Trainer    │      │ Experiment   │ │
│  │                  │      │                  │      │ Runner       │ │
│  │ - Load Dataset   │─────>│ - Train Model    │<─────│              │ │
│  │ - Preprocess     │      │ - Log Params     │      │ - Multiple   │ │
│  │ - Split Data     │      │ - Log Metrics    │      │   Runs       │ │
│  │ - Scale Features │      │ - Save Artifacts │      │ - Compare    │ │
│  └──────────────────┘      └────────┬─────────┘      └──────────────┘ │
│                                     │                                   │
│                                     │ MLflow API                        │
│                                     ▼                                   │
│                          ┌──────────────────────┐                      │
│                          │  MLflow Tracking     │                      │
│                          │  Server              │                      │
│                          │                      │                      │
│                          │  - Experiments       │                      │
│                          │  - Runs & Metrics    │                      │
│                          │  - Artifacts Store   │                      │
│                          │  - Model Registry    │                      │
│                          └──────────┬───────────┘                      │
│                                     │                                   │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
                                      │ Model Registry
                                      │ API
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Production Environment                            │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Docker Compose Orchestration                   │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────┐    ┌──────────────────────────┐   │  │
│  │  │  MLflow Server          │    │  Model Serving API       │   │  │
│  │  │  Container              │    │  Container               │   │  │
│  │  │                         │    │                          │   │  │
│  │  │  Port: 5000            │◄───┤  - Load Model from       │   │  │
│  │  │                         │    │    Registry              │   │  │
│  │  │  - Backend Store (DB)   │    │  - Load Scaler           │   │  │
│  │  │  - Artifact Store       │    │  - FastAPI Server        │   │  │
│  │  │  - Model Registry       │    │  - /health endpoint      │   │  │
│  │  │                         │    │  - /predict endpoint     │   │  │
│  │  └─────────────────────────┘    │                          │   │  │
│  │                                  │  Port: 8000              │   │  │
│  │                                  └──────────┬───────────────┘   │  │
│  └─────────────────────────────────────────────┼───────────────────┘  │
│                                                 │                       │
└─────────────────────────────────────────────────┼───────────────────────┘
                                                  │
                                                  │ HTTP/JSON
                                                  ▼
                                          ┌───────────────┐
                                          │  API Clients  │
                                          │               │
                                          │  - Web Apps   │
                                          │  - Mobile     │
                                          │  - Services   │
                                          └───────────────┘
```

## Component Descriptions

### 1. Data Processor (`src/data_processor.py`)

**Purpose:** Handles data loading and preprocessing

**Responsibilities:**
- Load datasets from scikit-learn (Iris, Wine, Breast Cancer)
- Handle missing values and data validation
- Apply feature scaling using StandardScaler
- Split data into train/test sets with stratification
- Maintain reproducibility with fixed random states

**Outputs:**
- Scaled training data
- Scaled testing data
- Fitted scaler object for production use

### 2. Model Trainer (`src/model_trainer.py`)

**Purpose:** Core ML training pipeline with MLflow integration

**Responsibilities:**
- Initialize and train scikit-learn classification models
- Log hyperparameters to MLflow (C, penalty, dataset_name)
- Log evaluation metrics (accuracy, precision, recall, F1-score)
- Generate and log artifacts:
  - Trained model (as MLflow sklearn model)
  - Confusion matrix visualization
  - Classification report
  - Preprocessor scaler
- Register models to MLflow Model Registry with metadata
- Add model descriptions and tags for versioning

**MLflow Integration:**
- Wraps training in `mlflow.start_run()` context
- Uses explicit logging (no autolog) for fine-grained control
- Stores artifacts in organized structure
- Enables experiment comparison through UI

### 3. Experiment Runner (`run_experiments.py`)

**Purpose:** Orchestrate multiple training runs for comparison

**Responsibilities:**
- Execute 4+ distinct experiments with varied hyperparameters
- Demonstrate experiment tracking capabilities
- Facilitate model comparison through MLflow UI
- Support hyperparameter tuning workflows

**Configuration:**
- Multiple C values (0.1, 1.0, 10.0)
- Different penalty types (l1, l2)
- Consistent dataset for fair comparison

### 4. Model Promotion Script (`promote_model.py`)

**Purpose:** Automate best model selection and staging

**Responsibilities:**
- Query all registered model versions
- Compare models based on F1-score (or specified metric)
- Archive current Production models
- Promote best-performing model to Production stage
- Maintain model lifecycle management

**Stage Transitions:**
- None → Production (initial promotion)
- Production → Archived (when superseded)

### 5. MLflow Tracking Server

**Purpose:** Centralized experiment tracking and model registry

**Components:**
- **Backend Store:** SQLite database for metadata
- **Artifact Store:** File system for model artifacts
- **Model Registry:** Version control for models
- **Web UI:** Visualization at port 5000

**Data Stored:**
- Experiment runs with timestamps
- Parameters and metrics for each run
- Artifacts (models, plots, reports, scalers)
- Model versions with stages and tags

### 6. Model Serving API (`src/inference_api.py`)

**Purpose:** Production-ready REST API for predictions

**Technology:** FastAPI framework

**Endpoints:**
- `GET /health` - Health check for monitoring
- `POST /predict` - Prediction endpoint accepting JSON features

**Features:**
- **Lifespan Event:** Loads model and scaler once at startup
- **Model Loading:** Fetches from MLflow Model Registry (Production stage)
- **Scaler Loading:** Loads preprocessing artifacts
- **Error Handling:** Validates input and returns appropriate HTTP codes
- **Response Format:** Returns both class labels and probabilities

**Performance Optimization:**
- Model cached in memory (loaded once)
- Async-capable endpoints
- Minimal processing overhead

## Data Flow

### Training Pipeline Flow

```
1. Dataset → Data Processor → Preprocessed Data
                                    ↓
2. Preprocessed Data → Model Trainer → Trained Model
                                           ↓
3. Trained Model → MLflow Tracking Server → Model Registry
                                                 ↓
4. Model Registry → Promote Script → Production Model
```

### Inference Pipeline Flow

```
1. API Startup → Load Model from Registry → Load Scaler
                                                  ↓
2. Client Request (JSON) → API Endpoint → Validate Input
                                               ↓
3. Validated Input → Apply Scaler → Transform Features
                                          ↓
4. Transformed Features → Model Predict → Predictions + Probabilities
                                                ↓
5. Predictions → JSON Response → Client
```

## Deployment Architecture

### Docker Compose Services

**Service 1: MLflow Server**
- Image: `mlflow/mlflow:latest`
- Port: 5000 (host) → 5000 (container)
- Volumes:
  - `./mlruns:/mlflow/artifacts` - Artifact storage
  - `./mlflow.db:/mlflow/mlflow.db` - Metadata database
- Command: MLflow server with file-based backend

**Service 2: Model API**
- Build: Custom Dockerfile
- Port: 8000 (host) → 8000 (container)
- Environment Variables:
  - `MLFLOW_TRACKING_URI=http://mlflow_server:5000`
  - `REGISTERED_MODEL_NAME=ClassificationModel`
  - `MODEL_STAGE=Production`
- Dependencies: Requires mlflow_server to be running
- Base Image: Python 3.10 slim

### Network Configuration

- Both services in the same Docker network
- Internal service-to-service communication via DNS
- MLflow Server accessible as `mlflow_server` from API container
- External access via localhost ports

## Technology Stack

### Core Technologies
- **Python 3.10+** - Primary language
- **scikit-learn** - ML model training
- **MLflow** - Experiment tracking and model registry
- **FastAPI** - API framework
- **Uvicorn** - ASGI server

### Supporting Libraries
- **NumPy/Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **joblib** - Model serialization
- **pytest** - Testing framework

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **SQLite** - MLflow backend store

## Security Considerations

1. **Environment Variables:** Sensitive configuration externalized
2. **Container Isolation:** Services run in isolated containers
3. **Minimal Base Images:** Reduced attack surface
4. **No Hardcoded Credentials:** Configuration via environment
5. **Input Validation:** API validates all incoming requests

## Scalability Considerations

### Current State (Single-Server)
- Suitable for development and small-scale production
- Single MLflow server instance
- Single API instance

### Future Enhancements
- **Horizontal Scaling:** Multiple API instances behind load balancer
- **Database Backend:** PostgreSQL/MySQL for MLflow backend
- **Object Storage:** S3/Azure Blob for artifact storage
- **Model Caching:** Redis for frequently accessed models
- **Monitoring:** Prometheus + Grafana for metrics
- **Logging:** ELK/EFK stack for centralized logging

## Testing Strategy

### Unit Tests (`tests/test_api.py`)
- Mock MLflow model and scaler loading
- Test prediction endpoint with valid inputs
- Test error handling for invalid inputs
- Test health endpoint
- Ensure proper HTTP status codes

### Integration Testing (Manual)
1. Start services with `docker-compose up`
2. Run experiments with `run_experiments.py`
3. Promote best model with `promote_model.py`
4. Test API endpoints with curl/Postman
5. Verify MLflow UI shows all runs

## Monitoring and Observability

### Available Metrics
- MLflow UI: Experiment metrics and model performance
- API Health Endpoint: Service availability
- Docker Logs: Application logs and errors

### Recommended Additions
- Application Performance Monitoring (APM)
- Prediction latency tracking
- Model drift detection
- Resource utilization monitoring

## Maintenance and Operations

### Model Updates
1. Train new models with updated data
2. Compare performance in MLflow UI
3. Promote better model to Production
4. API automatically uses new model (restart required)

### Version Control
- Code: Git repository
- Models: MLflow Model Registry
- Artifacts: Versioned with MLflow runs
- Containers: Tagged Docker images

---

**Note:** This architecture is designed for development and demonstration. Production deployments should include additional considerations for security, scalability, monitoring, and disaster recovery.
