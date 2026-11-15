# 설계와 해석 범위

## 데이터

모든 입력은 `cx_research_lab/fixtures.py`에서 생성합니다. 분류 점수는 사람이 정한 값이고, NPS는 두 집단의 카운트입니다. 실험 관측값은 고정된 산식으로 생성합니다. 원문 수집, 모델 학습, 외부 API 연결은 없습니다. 시점은 실습용 정수로 표현합니다.

## R1 · 분류와 검토

채널별 보정 자료에서 `1 - p(true label)`의 유한 표본 보정 분위수를 구하고, 해당 기준을 통과하는 후보 집합을 만듭니다. 학습·보정·추론 ID의 중복, 보정 이후 추론 여부, 모델·분류체계 버전, 점수 누락을 검사합니다.

단일 후보에 추가 위험 조건이 없을 때만 `AUTO_DRAFT`를 반환합니다. 나머지는 검토 사유와 함께 `REVIEW`로 남습니다. 검토 용량을 넘으면 대기열로 이동하며 자동 초안으로 승격하지 않습니다. `manual_only`는 모든 항목을 검토로 보냅니다.

Split conformal의 coverage 해석에는 보정 자료와 추론 자료의 교환가능성 등 전제가 필요합니다. 채널·시간·버전 검사는 그 전제를 증명하지 않습니다. 이 작은 fixture는 일부러 오분류와 불완전한 coverage를 포함하므로 경험적 성능 검증 자료로 사용할 수 없습니다. 선택된 자동 초안의 정확도도 coverage와 별개입니다.

## R2 · NPS와 구성 효과

집단별 NPS를 `s`, 시점별 응답 비중을 `w`로 두면 다음 대칭 분해를 사용합니다.

```text
집단 내부 변화 = Σ ((w_after + w_before) / 2) × (s_after - s_before)
응답 구성 효과 = Σ ((s_after + s_before) / 2) × (w_after - w_before)
원 NPS 변화    = 집단 내부 변화 + 응답 구성 효과
```

표준화 점수는 별도로 고정한 모집단 비중을 사용합니다. 응답 수 ≤ 초대 수 ≤ 모집단 수를 검사하고 기준 모집단의 버전·해시를 기록합니다. 필수 집단의 응답이 없으면 남은 집단만으로 재정규화하지 않고 표준화 결과를 `null`, 상태를 `HOLD`로 반환합니다.

가중치가 비응답 편향을 해결했다는 뜻은 아닙니다. 집단 내 응답자가 비응답자를 대표한다는 가정은 검증하지 않습니다. 분해는 기술통계이며 인과효과를 식별하지 않습니다.

## R3 · 실험 판독

대상 집합, 배정 seed, 시작·종료 시점, 지표·CUPED 버전을 고정합니다. 단위별 해시를 이용한 1:1 Bernoulli 배정 후, 고정 시점에서 배정 누락·배정 일치 여부·SRM·결과 누락·오류율·사전 변수 시점을 먼저 검사합니다.

CUPED 계수는 별도의 과거 합성 표본으로 적합하고 실험 전에 고정합니다. `y_adjusted = y - theta × (x - historical_mean)`으로 조정한 뒤 원 효과와 조정 효과의 처리군−대조군 차이, 표준오차를 비교합니다. 기본 fixture는 처리군에 −4를 더하도록 생성한 예제입니다.

정상 입력은 `READOUT_READY`, 미종료 실험은 `NOT_READY`, 무결성 실패는 `HOLD`입니다. 판독 가능한 상태가 적용 승인이나 효과 보장을 뜻하지 않습니다. 구간은 정규근사이며 순차 검정·다중 지표 보정·군집 배정·처치 간 간섭은 다루지 않습니다.

## 공통 결과 보존

입력 계약을 위반하면 `ContractError`로 거부합니다. 유효한 입력의 검토·보류는 구조화된 결과로 남깁니다. 내용 해시로 결과 파일명을 정하고, 같은 내용의 재저장은 허용하되 같은 경로에 다른 내용이 있으면 거부합니다. 로컬 원자적 쓰기를 대상으로 하며 분산 저장소의 동시성·보존 정책은 포함하지 않습니다.

## 참고 자료

- Angelopoulos & Bates, [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)
- Deng et al., [Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data](https://ai.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)

위 자료는 방법의 배경입니다. 이 저장소의 구현 범위와 합성 결과는 코드·테스트를 기준으로 확인합니다.
