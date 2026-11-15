# CX Decision Lab

합성 데이터로 분류 결과의 검토 경계, NPS 구성 효과, 실험 판독 조건을 살펴보는 Python 실습입니다.
숫자를 계산하는 코드와 그 숫자를 해석해도 되는지 판단하는 코드를 함께 다룹니다.

외부 패키지나 API 키 없이 실행됩니다. 입력은 직접 만든 점수·집계·실험 관측값이며, 실제 응답이나 학습된 분류 모델을 포함하지 않습니다.

## 실험 세 가지

| 모듈 | 살펴보는 문제 | 결과 |
| --- | --- | --- |
| [분류와 검토 라우팅](cx_research_lab/conformal.py) | 후보가 하나여도 자동 처리해도 될까? | 채널별 보정, 위험·버전 검사, `AUTO_DRAFT` / `REVIEW`, 검토 대기열 |
| [NPS 변화 분해](cx_research_lab/nps.py) | 점수가 오른 이유가 응답자 구성 변화라면? | 원 NPS·고정 모집단 NPS, 집단 내부 변화·구성 효과, 누락 시 `HOLD` |
| [실험 무결성 검사](cx_research_lab/experiment.py) | 관측이 빠진 실험에서 효과를 읽어도 될까? | 배정·시점·SRM 검사, 원 효과·CUPED 비교, `HOLD` / `NOT_READY` |

## 실행

Python 3.12에서 검증했습니다.

```bash
git clone https://github.com/jiyongKim615/cx-decision-lab.git
cd cx-decision-lab
make lab
python3 -m cx_research_lab --project R1
python3 -m cx_research_lab --project R2
python3 -m cx_research_lab --project R3 --drop-outcome
```

`make lab`은 57개 회귀 테스트, 문법 검사, 정상·실패 시나리오 평가를 실행합니다.
평가 결과는 `artifacts/evaluation.json`에 생성됩니다. 저장된 [예제 결과](examples/evaluation.json)와 비교할 수 있습니다.

```bash
# 검토 인원을 1명으로 줄여도 대기 항목은 자동 초안으로 바뀌지 않습니다.
python3 -m cx_research_lab --project R1 --review-capacity 1

# 전체 수동 검토 모드와 내용 해시 기반 결과 보존
python3 -m cx_research_lab --manual-only --output-dir artifacts/runs
```

## 기본 합성 입력에서 관찰한 결과

| 시나리오 | 관찰값 |
| --- | --- |
| 분류 8건 | 자동 초안 2건, 검토 6건, 그중 대기 3건 |
| 자동 초안의 오분류 | 2건 중 1건 — 단일 후보가 정답을 보장하지 않는 예시 |
| NPS 변화 | 원 점수 −4 → 44, 내부 변화 0, 구성 효과 +48, 표준화 변화 0 |
| 필수 집단 응답 누락 | `HOLD`, 표준화 점수·변화는 `null` |
| 실험 400건 | 원 효과 −4.007, CUPED 효과 약 −3.874; 표준오차 약 1.423 → 0.201 |
| 실험 결과 1건 누락 | `HOLD`, 효과 계산 보류 |

이 수치는 [고정 fixture](cx_research_lab/fixtures.py)의 동작 확인값입니다. 분류 성능이나 실제 개선 효과의 추정값으로 사용할 수 없습니다.

## 읽는 순서

1. [설계와 해석 범위](docs/design.md): 입력 계약, 계산식, 검토·보류 조건
2. [평가와 실패 사례](docs/evaluation.md): 재현 방법과 테스트 범위
3. [실행 예제](cx_research_lab/demo.py) → [회귀 테스트](tests/test_decisions.py)

계산 결과에는 버전·입력 해시·실행 trace를 남깁니다. 파일 보존은 로컬 파일시스템을 대상으로 하며, 서비스 배포나 자동 실행을 포함하지 않습니다.
