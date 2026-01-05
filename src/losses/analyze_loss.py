"""
Ultralytics v8DetectionLoss가 기대하는 batch 형식 분석
"""
import inspect

try:
    from ultralytics.utils.loss import v8DetectionLoss
    
    print("=" * 70)
    print("v8DetectionLoss.__call__ 소스코드 분석")
    print("=" * 70)
    
    # __call__ 소스코드 확인
    source = inspect.getsource(v8DetectionLoss.__call__)
    
    print("\n[batch 관련 코드 라인]")
    lines = source.split('\n')
    for i, line in enumerate(lines):
        # batch. 또는 batch[ 패턴 찾기
        if 'batch' in line and ('batch.' in line or 'batch[' in line):
            print(f"  Line {i}: {line.strip()}")
    
    print("\n" + "=" * 70)
    print("전체 __call__ 소스:")
    print("=" * 70)
    print(source[:3000])  # 처음 3000자만
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()