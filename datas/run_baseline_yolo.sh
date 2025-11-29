#!/bin/bash

# =================================================================
#  Project AIS-MAN: YOLO Baseline Automation Script (RTX 4090 Edition)
#  Author: Changmin Lee & AI Partner
#  Date: 2025-11-26 (Updated: Directory Structure Fix)
# =================================================================

# [중요] 경로 설정 (Path Configuration)
# 스크립트가 어디서 실행되든 상관없도록 절대 경로를 사용함
BASE_DIR="/home/octolab-rtx4090/Desktop/changmin/dark_vessel_sr_yolo"
SAVE_DIR="$BASE_DIR/yolo_results"

# 1. 하이퍼파라미터 설정 (Hyperparameters)
DATA_YAML="/home/octolab-rtx4090/Desktop/changmin/airbus_data/data.yaml"
EPOCHS=100
IMG_SIZE=192
BATCH_SIZE=512
WORKERS=8
DEVICE=0

# 2. 학습할 모델 리스트 (Models to Train)
MODELS=("yolov8n.pt" "yolov8s.pt" "yolo11n.pt" "yolo11s.pt")

echo "========================================================"
echo "🚀 AIS-MAN Baseline Training Started on RTX 4090"
echo "📍 Base Directory: $BASE_DIR"
echo "💾 Output Directory: $SAVE_DIR"
echo "🎯 Total Models: ${#MODELS[@]}"
echo "========================================================"

# 결과 저장용 폴더 생성 (없으면 생성)
mkdir -p "$SAVE_DIR"

# 3. 반복 루프 (Training Loop)
for MODEL in "${MODELS[@]}"
do
    # 모델 파일명에서 .pt 확장자 제거 (예: yolov8n.pt -> yolov8n)
    MODEL_NAME="${MODEL%.*}"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "▶️  Processing Model: $MODEL_NAME (Start Time: $(date))"
    echo "--------------------------------------------------------"

    # [핵심 수정 사항]
    # project: 상위 폴더 (yolo_results)
    # name: 모델별 폴더 이름 (예: yolov8n) -> 이렇게 하면 yolo_results/yolov8n/weights/... 로 저장됨
    
    yolo detect train \
        project="$SAVE_DIR" \
        name="$MODEL_NAME" \
        model="$MODEL" \
        data="$DATA_YAML" \
        epochs=$EPOCHS \
        imgsz=$IMG_SIZE \
        batch=$BATCH_SIZE \
        device=$DEVICE \
        workers=$WORKERS \
        patience=10 \
        save=True \
        exist_ok=True \
        pretrained=True \
        optimizer='auto' \
        verbose=True \
        val=True \
        cache=True \
        amp=True 

    # 학습 완료 후 메시지
    if [ $? -eq 0 ]; then
        echo "✅ [SUCCESS] Training finished for $MODEL_NAME"
        echo "📂 Results saved at: $SAVE_DIR/$MODEL_NAME"
    else
        echo "❌ [FAILURE] Error occurred while training $MODEL_NAME"
    fi

done

echo ""
echo "========================================================"
echo "🎉 All Jobs Completed! (End Time: $(date))"
echo "📂 Final Check: ls -F $SAVE_DIR"
ls -F "$SAVE_DIR"
echo "========================================================"