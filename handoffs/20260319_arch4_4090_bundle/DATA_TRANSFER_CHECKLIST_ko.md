# 4090 PC 데이터 전달 체크리스트

## 꼭 필요한 것
- 프로젝트 폴더
  - `/home/changmin/dark_vessel_sr_yolo`
- LR 데이터
  - `/home/changmin/smart_airbus_data_lr`
- HR 데이터
  - `/home/changmin/smart_airbus_data`

## 가능하면 안 옮겨도 되는 것
- Sniper crop dataset
  - `/home/changmin/dark_vessel_sr_yolo/data/arch4_sniper_crops`

이 폴더는 용량이 크고 복사 오류가 나기 쉬우므로, 가능하면:
- raw LR/HR 데이터만 옮기고
- 4090에서 `REBUILD_SNIPER_CROPS_ON_4090.sh`로 다시 만드는 것을 권장

## 최소 실행 기준

### 1. Scout 진단만 할 경우
- 필요:
  - project
  - `smart_airbus_data_lr`
  - `smart_airbus_data`

### 2. Arch4 full-val 평가만 할 경우
- 필요:
  - project
  - `smart_airbus_data_lr`
  - `smart_airbus_data`

### 3. Sniper hard-negative / crop 재학습까지 할 경우
- 필요:
  - project
  - `smart_airbus_data_lr`
  - `smart_airbus_data`
  - 그리고 둘 중 하나:
    - `data/arch4_sniper_crops`
    - 또는 4090에서 crop dataset 재생성

## 권장
- 경로를 원본과 동일하게 맞춘다.
- 경로가 달라지면 절대경로가 들어간 yaml과 shell script를 다시 패치해야 한다.
