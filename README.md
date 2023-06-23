# 📃 프로젝트 개요

  <p align="center"><img src="https://github.com/SS-hj/DACON-objectdetection/assets/54202082/b40140a1-62cc-40a3-abd4-74a560de8302" width="100%"/></p>

합성데이터란 실제 환경에서 수집되거나 측정되는 것이 아니라 디지털 환경에서 생성되는 데이터셋으로, 최근 방대한 양질의 데이터셋이 필요해짐에 따라 그 중요성이 대두되고 있습니다.
이러한 합성데이터를 활용하여 자동차 탐지를 수행하여, 34가지의 자동차 세부모델까지 판별하는 AI 모델을 개발하는 대회입니다.

<br/>

# 💾 데이터셋

- 전체 이미지 개수 : 9754장 (Training : 6481장, Test : 1700장)
- 34 class : chevrolet_malibu_sedan_2012_2016, chevrolet_malibu_sedan_2017_2019, ..., ssangyong_tivoli_suv_2016_2020
- 이미지 크기 : (1920, 1040)

<br/>

# ✏ 프로젝트 수행 방법

  <br/>

## Data Preprocessing

- StratifiedGroupKFold
- Heavy Augmentation
- Validation Augmentation for Synthetic data
  ```
  💡 validation aug를 주어 실제 test data와 유사한 노이즈를 가지도록 하기
  ```
  - 문제상황 : 합성 데이터로 train과 validation을 나누어 진행하여, test 데이터와 차이 발생
      1. 합성 데이터 그대로 학습을 진행하게 되면 2 에포크 정도만 돌아도 모든 score가 만점을 받게 된다.
      2. 이렇게 만점을 받아도 real 데이터인 test 데이터에서의 점수는 높지 않다.
  - 아이디어 : validation과 test 데이터와의 간극을 좁힐 필요가 있다.
      1. 실제 데이터에서는 움직이고 있는 blur 된 차량 이미지들이 있다. ⇒ Blur 관련 aug를 추가
      2. 실제 데이터에서는 가로등에 의해 일부분이 가려진 차량 이미지들이 있다. ⇒ CoarseDropout aug를 추가
      3. 차량의 색상은 class label에 영향을 끼치지 않는다. ⇒ 색상 관련 aug를 추가
 

  <br/>

## Modeling

- Model
    - Cascade RCNN
    - Cascade Mask RCNN
- Backbone
  - Swin Transformer - Small
  - ConvNeXt - Tiny
  - Res2Net

<br/>

# 🏆 프로젝트 결과
- Ensemble

  <p align="center"><img src="https://github.com/SS-hj/DACON-objectdetection/assets/54202082/aac2ca29-e941-414e-a6fb-933ca1bdf2cc" alt="trash" width="50%" height="50%" /></p>


<table align="center">
  <tr height="35px">
    <td align="center" width="180px">
      <a> convnext </a>
    </td>
    <td align="center" width="180px">
      <a> swin </a>
    </td>
    <td align="center" width="180px">
      <a> res2net </a>
    </td>
    <td align="center" width="180px">
      <a> Ensemble </a>
    </td>
  </tr>
  <tr height="35px">
    <td align="center" width="180px">
      <a> 0.954 </a>
    </td>
    <td align="center" width="180px">
      <a> 0.963 </a>
    </td>
    <td align="center" width="180px">
      <a> 0.955 </a>
    </td>
    <td align="center" width="180px">
      <a> 0.992 </a>
    </td>
  </tr>
</table>


- Score

  <p align="center"><img src="https://github.com/SS-hj/DACON-objectdetection/assets/54202082/6969b445-9488-446d-b757-72741ae1ef3f" width="100%" height="100%" /></p>

<br/>


# 팀원 소개

<table align="center">
  <tr height="35px">
    <td align="center" width="320px">
      <a href="https://github.com/soonyoung-hwang">황순영</a>
    </td>
    <td align="center" width="320px">
      <a href="https://github.com/SS-hj">이하정</a>
    </td>
  </tr>
  <tr height="35px">
    <td align="center" width="320px">
      <a> swin, res2net modeling, ensemble </a>
    </td>
    <td align="center" width="320px">
      <a> data preprocessing, convnext modeling </a>
    </td>
  </tr>
</table>