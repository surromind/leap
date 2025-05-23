## Model
**Tip**: Hugging Face Hub에서 제공하며, vLLM이 지원하는 LLM을 사용할 수 있습니다.

- **Baseline**: 베이스라인 모델 목록을 확인할 수 있습니다.
- **Finetuned**: 해당 시스템에서 튜닝한 모델 목록을 확인할 수 있습니다.
- **Custom**: 선택 목록에 없는 모델을 사용하려면 **Custom Model** 텍스트 박스에 원하는 모델 이름을 입력하세요.
    - 예시: `google/gemma-1.1-2b-it`
    - Custom 모델 로드 실패 시 다음 항목을 확인하세요:
        1. Hugging Face Hub 토큰이 유효한지 확인하세요.
        2. 해당 토큰을 발급받은 계정으로 해당 모델에 대한 접근 권한이 있는지 확인하세요.
        3. 모델이 vLLM에서 지원되는지 확인하세요. 원하는 모델이 지원 목록에 없을 경우, vLLM을 업그레이드 해보세요.<br>
        [지원 모델 목록 확인하기](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html)



## Dataset
**Tip**: 모델 성능 평가를 위한 1개 이상의 데이터셋을 선택합니다. 
- **Evaluation Dataset** 탭에서 자세한 데이터셋 정보를 확인할 수 있습니다.


## Device
**Tip**: 사용 중이지 않은 GPU를 우선적으로 선택하세요. 
- **사용 가능한 GPU의 개수, 종류, 및 현재 상태는 Device 체크박스와 그 아래 상태 표시 바를 통해 확인할 수 있습니다.** 
- Tensor Parallel Size와 동일한 개수의 GPU를 선택해야 최적의 성능을 낼 수 있습니다.
- 최신 GPU는 더 큰 메모리 용량과 높은 연산 성능을 제공하므로, 고사양 GPU를 우선적으로 선택하는 것이 효율적입니다.

## Tensor Parallel Size
**Tip**: tensor_parallel_size = **GPU 수**로 설정하는 것이 좋습니다.
- infer backend가 vllm인 경우만 2 이상 설정 가능합니다. 
- vllm 특성상 16의 약수 (1, 2, 4, 8, 16)로 설정해야 합니다.

## GPU Memory Utilization
**Tip**: 권장 값은 **0.8~0.9** 사이입니다.
- 모델 실행에 사용할 GPU 메모리 비율을 설정하며, 0에서 1 사이 값으로 지정할 수 있습니다.
- 다른 프로세스에서 이미 GPU 메모리를 사용 중이라면, 남은 메모리 비율에 따라 실제 활용 가능한 비율이 줄어들 수 있습니다.
- 사용 중인 프로세스를 고려하여 GPU 메모리 활용률을 조정하고, **너무 높은 값을 설정하지 않아 OOM(Out of Memory) 오류를 방지하세요.**

## Max Model Length
**Tip**: 현재 설정 상, 모델 구성에서 자동으로 파생된 값을 **최댓값**으로 사용하고 있으므로 최댓값을 사용하는 것이 좋습니다.  
- 모델이 처리할 수 있는 최대 입력 시퀀스 길이이
- 사용할 모델이 지원하는 최대 토큰 길이를 확인한 뒤, 입력 시퀀스 길이를 고려하여 설정합니다. 
- 예: GPT-3 기반 모델은 보통 2048~4096 토큰까지 지원. 
- 작업에 필요한 맥락의 크기를 고려해 적절히 설정하면 메모리 사용량을 줄이고 효율성을 높일 수 있습니다.

## Max New Tokens
**Tip**: 생성할 토큰 수를 작업의 요구사항에 맞게 설정하세요.
- 필요 이상으로 토큰을 생성하지 않도록 설정하면 시간과 자원을 절약할 수 있습니다.
- 권장 값
    - 짧은 답변: 20~50
    - 중간 길이 답변: 100~200
    - 긴 생성 작업: 500~1000 
## Num Fewshot
**Tip**: Few-shot 학습 예제 수는 메모리 및 성능과 균형을 맞춥니다. 
- 작은 GPU 메모리: 1~2개
- 큰 GPU 메모리 및 높은 정확도 필요: 4~5개
## Top K
**Tip**: 생성 다양성을 조절하며, 낮은 값을 사용할수록 보수적으로 생성합니다.
- **권장 값: 40~100 (일반적으로 50이 적절)**
- 작은 작업(단답형 질문)은 낮은 값(10~20), 창의적인 작업은 높은 값(100 이상)
## Top P (Nucleus Sampling)
**Tip**: 확률 기반으로 토큰 선택 범위를 제한합니다.
- **권장 값: 0.8~0.95 (안정적이며 다양성 유지)**
- 생성이 불안정할 경우 값을 낮추고, 더 창의적이고 자유로운 생성을 원할 경우 값을 높임.
## Temperature
**Tip**: 모델의 출력을 더 무작위로 만들거나, 더 결정론적으로 만듭니다.
- 결정론적 생성: 0.1~0.3
- 균형 잡힌 설정: 0.7
- 창의적 생성: 1.0 이상

------------


## Inference Memory
총 추론 메모리는 모델 가중치를 저장하는 데 필요한 메모리와 순전파 중 약간의 추가 오버헤드를 고려하여 다음과 같이 계산됩니다. <br><br>
$\text{Total Memory}_{\text{Inference}} \approx 1.2 \times \text{Model Memory}$.

## Training Memory
총 학습 메모리는 모델 파라미터 메모리, 옵티마이저 상태 메모리, 활성화 메모리, 그리고 그래디언트 메모리의 합으로 계산됩니다.<br><br>
$\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Activation Memory} + \text{Gradient Memory}$<br><br>
각 구성 요소는 데이터 유형(fp32, fp16), 옵티마이저 유형(예: AdamW, 8비트), 활성화 재계산 전략(No/Selective/Full Recomputation) 등 여러 변수에 따라 다릅니다.<br>
아래 계산된 훈련 메모리는 해당 설정을 기반으로 계산하였습니다. 
- 데이터 유형: fp32
- 옵티마이저 유형: AdamW,
- 활성화 재계산 전략: Full Recomputation
- 배치 사이즈: 1
- 텐서 병렬화 사이즈: 1




