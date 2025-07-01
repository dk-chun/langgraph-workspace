# Basic Agent Tests

이 디렉토리에는 Basic Agent의 포괄적인 테스트가 포함되어 있습니다.

## 테스트 구조

### 테스트 클래스

1. **TestBasicAgentLocal**: 로컬 테스트 (서버 없이)
   - Agent 생성 및 설정
   - Factory 함수 테스트
   - 설정 상속 확인

2. **TestBasicAgentServer**: LangGraph 서버를 통한 테스트
   - 서버 상태 확인
   - 7개 템플릿 테스트 (chat, summarize, translate, explain, code, creative, analyze)
   - 커스텀 시스템 메시지
   - 성능 타이밍

3. **TestBasicAgentIntegration**: 통합 테스트
   - 다중 턴 대화
   - 복합 기능 테스트

### 테스트 템플릿

Basic Agent는 7개의 내장 템플릿을 지원합니다:

- `chat`: 일반 대화
- `summarize`: 텍스트 요약
- `translate`: 번역
- `explain`: 설명
- `code`: 코드 생성
- `creative`: 창작
- `analyze`: 분석

## 실행 방법

### 1. 간단한 실행

```bash
# 모든 테스트 실행
python run_tests.py

# 로컬 테스트만 실행
python run_tests.py --type local

# 서버 테스트만 실행
python run_tests.py --type server

# 통합 테스트만 실행
python run_tests.py --type integration
```

### 2. 의존성 설치와 함께 실행

```bash
python run_tests.py --install-deps --check-ollama
```

### 3. 직접 pytest 실행

```bash
# 테스트 의존성 설치
pip install -r tests/test_requirements.txt

# 모든 테스트 실행
pytest tests/ -v

# 특정 테스트 클래스만 실행
pytest tests/test_basic_agent.py::TestBasicAgentLocal -v

# 특정 템플릿 테스트만 실행
pytest tests/test_basic_agent.py::TestBasicAgentServer::test_all_templates -v
```

## 사전 요구사항

### Ollama 설정

서버 테스트를 실행하려면 Ollama가 실행 중이어야 합니다:

```bash
# Ollama 실행
ollama serve

# 필요한 모델 다운로드
ollama pull deepseek-r1:latest
```

### 환경 변수

`.env` 파일에 다음 설정이 필요합니다:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:latest
```

### LangGraph 서버

서버 테스트는 자동으로 LangGraph 서버를 시작/종료합니다. 수동으로 실행하려면:

```bash
langgraph dev --port 8123
```

## 테스트 마커

pytest 마커를 사용하여 특정 테스트만 실행할 수 있습니다:

```bash
# 로컬 테스트만
pytest -m local

# 서버 테스트만
pytest -m server

# 통합 테스트만
pytest -m integration

# 느린 테스트 제외
pytest -m "not slow"
```

## 트러블슈팅

### 일반적인 문제

1. **Ollama 연결 실패**
   ```
   ❌ Ollama server not available at http://localhost:11434
   ```
   - Ollama 서버가 실행 중인지 확인
   - 포트 설정 확인

2. **모델 없음**
   ```
   ❌ Required model deepseek-r1:latest not found in Ollama
   ```
   - `ollama pull deepseek-r1:latest` 실행

3. **LangGraph 서버 시작 실패**
   ```
   LangGraph server failed to start within 30 seconds
   ```
   - 포트 8123이 사용 중인지 확인
   - `langgraph.json` 설정 확인

4. **테스트 타임아웃**
   - Ollama 모델 응답이 느릴 수 있음
   - 더 빠른 모델 사용 고려

### 디버깅

테스트 실패 시 자세한 정보를 보려면:

```bash
pytest tests/ -v --tb=long --no-header
```

## 성능 고려사항

- **로컬 테스트**: 빠름 (1-2초)
- **서버 테스트**: 중간 (템플릿당 5-10초)
- **통합 테스트**: 느림 (10-30초)

전체 테스트 수행 시간: 약 2-5분 (Ollama 모델 속도에 따라) 